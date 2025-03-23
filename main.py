import requests
from bs4 import BeautifulSoup
import re
import json
import os
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor


class WikipediaCrawler:
    def __init__(self, start_url, max_pages=50, output_file="wikipedia_data.json", queue_file="remaining_links.json",
                 num_threads=5):
        self.start_url = start_url
        self.visited = set()
        self.to_visit = set()
        self.max_pages = max_pages
        self.output_file = output_file
        self.queue_file = queue_file
        self.num_threads = num_threads

        # Locks for thread safety
        self.visited_lock = threading.Lock()
        self.to_visit_lock = threading.Lock()
        self.json_lock = threading.Lock()

        # Stats tracking
        self.pages_crawled = 0
        self.pages_crawled_lock = threading.Lock()

        # Initialize or load existing data
        self.load_or_create_json_file()

        # Load or initialize the queue
        self.load_or_create_queue(start_url)

    def load_or_create_json_file(self):
        """Load existing JSON file or create a new one if it doesn't exist."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Add existing URLs to visited set
                    for entry in data:
                        if "wiki_url" in entry:
                            self.visited.add(entry["wiki_url"])
                print(f"Loaded existing data from {self.output_file}")
                print(f"Found {len(self.visited)} previously visited URLs")
            except json.JSONDecodeError:
                print(f"Error in existing JSON file, creating new file")
                # Create an empty array in the file
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
        else:
            print(f"Creating new JSON file: {self.output_file}")
            # Create an empty array in the file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def load_or_create_queue(self, start_url):
        """Load existing queue or create a new one with the start URL."""
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, 'r', encoding='utf-8') as f:
                    queue_data = json.load(f)
                    # Add links to the to-visit set
                    for url in queue_data:
                        if url not in self.visited:
                            self.to_visit.add(url)
                print(f"Loaded {len(self.to_visit)} links from queue file")
            except json.JSONDecodeError:
                print(f"Error in queue file, starting with initial URL")
                self.to_visit.add(start_url)
        else:
            print(f"Queue file does not exist, starting with initial URL")
            self.to_visit.add(start_url)

        # If there are no valid links in the queue (all visited), add the start URL
        if not self.to_visit and start_url not in self.visited:
            print("No valid links in queue, adding start URL")
            self.to_visit.add(start_url)

    def save_queue(self):
        """Save the current to-visit queue to the queue file."""
        with self.to_visit_lock:
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.to_visit), f, indent=2)
            to_visit_size = len(self.to_visit)
        print(f"Saved {to_visit_size} links to queue file")

    def fetch_page(self, url):
        """Fetches a Wikipedia page and returns its parsed soup."""
        try:
            headers = {
                'User-Agent': 'WikipediaCrawler/1.0 (Educational Research Project)'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def get_title(self, soup):
        """Extracts the title of the Wikipedia page."""
        try:
            title_element = soup.find('h1', {'id': 'firstHeading'})
            if title_element:
                return title_element.text.strip()
            return "Title not found"
        except Exception as e:
            return f"Error extracting title: {e}"

    def get_summary(self, soup):
        """Extracts the summary section from a Wikipedia page."""
        try:
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return "Content div not found on the page."

            # Find the first section - this is the content before the first heading
            first_section_paragraphs = []

            # Start from the first paragraph
            current_element = content_div.find('p')

            # Continue until we hit a heading (h2, h3, etc.) or run out of elements
            while current_element:
                # If we encounter a heading, we've reached the end of the first section
                if current_element.name and current_element.name.startswith('h'):
                    break

                # If it's a paragraph, add its text to our collection
                if current_element.name == 'p' and current_element.text.strip():
                    # Remove citation numbers [1], [2], etc.
                    clean_text = re.sub(r'\[\d+\]', '', current_element.text)
                    first_section_paragraphs.append(clean_text.strip())

                # Move to the next element
                current_element = current_element.find_next_sibling()

            # Join all paragraphs with newlines in between
            if first_section_paragraphs:
                return '\n\n'.join(first_section_paragraphs)
            else:
                return "No paragraphs found in the first section."

        except Exception as e:
            return f"An error occurred while extracting summary: {e}"

    def extract_links(self, soup):
        """Extracts Wikipedia article links from a BeautifulSoup object."""
        links = set()
        for link in soup.select('a[href^="/wiki/"]'):
            href = link.get('href')
            if ':' not in href and '#' not in href:  # Filter out non-article pages
                full_url = f"https://en.wikipedia.org{href}"
                links.add(full_url)
        return links

    def append_to_json(self, title, summary, url):
        """Appends new data to the existing JSON file."""
        with self.json_lock:
            # Read current data
            with open(self.output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if URL already exists
            for entry in data:
                if entry.get("wiki_url") == url:
                    print(f"URL {url} already exists in the JSON file, skipping")
                    return False

            # Add new entry
            new_entry = {
                "title": title,
                "summary": summary,
                "wiki_url": url
            }
            data.append(new_entry)

            # Write updated data back to file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return True

    def crawl_url(self, url):
        """Process a single URL and return any new links found."""
        thread_name = threading.current_thread().name
        print(f"\n{thread_name} - Crawling: {url}")

        # Check if already visited (thread safe)
        with self.visited_lock:
            if url in self.visited:
                print(f"{thread_name} - URL {url} already visited, skipping")
                return set()
            # Mark as visited
            self.visited.add(url)

        # Fetch the page content
        soup = self.fetch_page(url)
        if not soup:
            return set()

        # Extract the title
        title = self.get_title(soup)
        print(f"{thread_name} - TITLE: {title}")

        # Extract the summary
        summary = self.get_summary(soup)
        print(f"{thread_name} - Extracted summary")

        # Append to JSON file
        if self.append_to_json(title, summary, url):
            with self.pages_crawled_lock:
                self.pages_crawled += 1
                current_count = self.pages_crawled
            print(f"{thread_name} - Added to JSON. Pages crawled: {current_count}")

        # Extract links
        new_links = self.extract_links(soup)
        print(f"{thread_name} - Found {len(new_links)} links")

        return new_links

    def migrate_and_crawl_links_threaded(self):
        """
        Multi-threaded version of migrate_and_crawl_links
        """
        original_queue_file = self.queue_file
        new_queue_file = "remaining_links_2.json"

        # Keep track of the original to_visit set
        with self.to_visit_lock:
            original_to_visit = list(self.to_visit.copy())
            self.to_visit = set()  # Clear for new links

        # Create empty new queue file
        with open(new_queue_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

        print(f"\n{'=' * 80}")
        print(f"Starting multi-threaded migration process from {original_queue_file} to {new_queue_file}")
        print(f"Total links to crawl: {len(original_to_visit)}")
        print(f"Using {self.num_threads} threads")
        print(f"{'=' * 80}\n")

        # Set new queue file for saving newly discovered links
        self.queue_file = new_queue_file

        # Reset pages crawled counter
        self.pages_crawled = 0

        # Create a work queue
        work_queue = queue.Queue()
        for url in original_to_visit:
            work_queue.put(url)

        # Flag to signal when all work is done
        all_done = threading.Event()

        # Flag to control periodic queue saving
        save_interval = 10  # seconds
        last_save_time = time.time()

        def worker():
            while not all_done.is_set():
                try:
                    # Get a URL with a 1 second timeout
                    url = work_queue.get(timeout=1)

                    # Process the URL
                    new_links = self.crawl_url(url)

                    # Add any new links to the to_visit set
                    with self.to_visit_lock:
                        for link in new_links:
                            with self.visited_lock:
                                if link not in self.visited:
                                    self.to_visit.add(link)

                    # Mark task as done
                    work_queue.task_done()
                except queue.Empty:
                    # No more work in the queue
                    time.sleep(0.1)  # Small sleep to prevent CPU spinning

        # Start worker threads
        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=worker, name=f"Worker-{i + 1}")
            t.daemon = True
            t.start()
            threads.append(t)

        try:
            # Monitor progress and save queue periodically
            while not work_queue.empty() or self.pages_crawled < len(original_to_visit):
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    self.save_queue()
                    last_save_time = current_time

                    # Print status
                    with self.pages_crawled_lock:
                        current_count = self.pages_crawled
                    print(f"\nStatus: {current_count}/{len(original_to_visit)} pages processed")
                    print(f"Queue size: {work_queue.qsize()}")

                time.sleep(1)

            # Wait for all tasks to be processed
            work_queue.join()

        except KeyboardInterrupt:
            print("\nCrawling interrupted by user")
        finally:
            # Signal all workers to exit
            all_done.set()

            # Wait for all threads to finish
            for t in threads:
                t.join(timeout=2)

            # Save the new queue before exiting
            self.save_queue()

            print(f"\n{'=' * 80}")
            print(f"Migration process complete.")
            print(f"Processed {self.pages_crawled} pages from original queue.")

            with self.to_visit_lock:
                new_links_count = len(self.to_visit)
            print(f"Found {new_links_count} new links saved to {new_queue_file}")

            # Replace the original queue file with the new one
            if os.path.exists(original_queue_file):
                os.remove(original_queue_file)
                print(f"Deleted original queue file: {original_queue_file}")

            os.rename(new_queue_file, original_queue_file)
            print(f"Renamed {new_queue_file} to {original_queue_file}")

            # Update the queue file name back to the original
            self.queue_file = original_queue_file

            print(f"Data saved to {self.output_file}")
            print(f"Queue migration completed successfully.")
            print(f"{'=' * 80}\n")

    def migrate_and_crawl_links_with_threadpool(self):
        """
        Alternative implementation using ThreadPoolExecutor for simpler thread management
        """
        original_queue_file = self.queue_file
        new_queue_file = "remaining_links_2.json"

        # Keep track of the original to_visit set
        with self.to_visit_lock:
            original_to_visit = list(self.to_visit.copy())
            self.to_visit = set()  # Clear for new links

        # Create empty new queue file
        with open(new_queue_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

        print(f"\n{'=' * 80}")
        print(f"Starting ThreadPool migration process from {original_queue_file} to {new_queue_file}")
        print(f"Total links to crawl: {len(original_to_visit)}")
        print(f"Using {self.num_threads} threads")
        print(f"{'=' * 80}\n")

        # Set new queue file for saving newly discovered links
        self.queue_file = new_queue_file

        # Reset pages crawled counter
        self.pages_crawled = 0

        # Set up queue saving interval
        save_interval = 10  # seconds
        last_save_time = time.time()

        try:
            # Process URLs with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all URLs to the executor
                future_to_url = {executor.submit(self.crawl_url, url): url for url in original_to_visit}

                # Process as they complete
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        # Get new links from the completed task
                        new_links = future.result()

                        # Add new links to the to_visit set
                        with self.to_visit_lock:
                            for link in new_links:
                                with self.visited_lock:
                                    if link not in self.visited:
                                        self.to_visit.add(link)

                        # Save queue periodically
                        current_time = time.time()
                        if current_time - last_save_time > save_interval:
                            self.save_queue()
                            last_save_time = current_time

                            # Print status
                            with self.pages_crawled_lock:
                                print(f"\nStatus: {self.pages_crawled}/{len(original_to_visit)} pages processed")

                    except Exception as e:
                        print(f"Error processing {url}: {e}")

        except KeyboardInterrupt:
            print("\nCrawling interrupted by user")
        finally:
            # Save the new queue before exiting
            self.save_queue()

            print(f"\n{'=' * 80}")
            print(f"Migration process complete.")
            print(f"Processed {self.pages_crawled} pages from original queue.")

            with self.to_visit_lock:
                new_links_count = len(self.to_visit)
            print(f"Found {new_links_count} new links saved to {new_queue_file}")

            # Replace the original queue file with the new one
            if os.path.exists(original_queue_file):
                os.remove(original_queue_file)
                print(f"Deleted original queue file: {original_queue_file}")

            os.rename(new_queue_file, original_queue_file)
            print(f"Renamed {new_queue_file} to {original_queue_file}")

            # Update the queue file name back to the original
            self.queue_file = original_queue_file

            print(f"Data saved to {self.output_file}")
            print(f"Queue migration completed successfully.")
            print(f"{'=' * 80}\n")

    # Original methods remain the same
    def crawl(self):
        """Original crawl method - preserved for backward compatibility"""
        pages_crawled = 0

        try:
            while self.to_visit and pages_crawled < self.max_pages:
                # Get the next URL to visit
                url = self.to_visit.pop()

                print(f"\n{'=' * 80}\nCrawling: {url}\n{'=' * 80}")

                # Skip if already visited
                if url in self.visited:
                    print(f"URL {url} already visited, skipping")
                    continue

                # Mark as visited
                self.visited.add(url)

                # Fetch the page content
                soup = self.fetch_page(url)

                if soup:
                    # Extract the title
                    title = self.get_title(soup)
                    print(f"TITLE: {title}")

                    # Extract the summary
                    summary = self.get_summary(soup)
                    print("\nSUMMARY:")
                    print(summary)

                    # Append to JSON file
                    if self.append_to_json(title, summary, url):
                        print(f"\nAppended to {self.output_file}")
                        pages_crawled += 1
                        print(f"Pages crawled: {pages_crawled}/{self.max_pages}")

                    # Extract links
                    print("\nExtracting links...")
                    new_links = self.extract_links(soup)
                    print(f"Found {len(new_links)} links")

                    # Add new links to the to_visit set
                    for link in new_links:
                        if link not in self.visited and link not in self.to_visit:
                            self.to_visit.add(link)

                    # Periodically save the queue (every page)
                    self.save_queue()

        except KeyboardInterrupt:
            print("\nCrawling interrupted by user")
        finally:
            # Save the queue before exiting
            self.save_queue()
            print(f"Crawling finished. Processed {pages_crawled} pages.")
            print(f"Data saved to {self.output_file}")
            print(f"Remaining links saved to {self.queue_file}")
            print(f"{len(self.to_visit)} links remaining in the queue")

    def migrate_and_crawl_links(self):
        """Original migrate_and_crawl_links method - preserved for backward compatibility"""
        original_queue_file = self.queue_file
        new_queue_file = "remaining_links_2.json"

        # Keep track of the original to_visit set
        original_to_visit = self.to_visit.copy()

        # Create empty new queue file
        with open(new_queue_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

        print(f"\n{'=' * 80}")
        print(f"Starting migration process from {original_queue_file} to {new_queue_file}")
        print(f"Total links to crawl: {len(original_to_visit)}")
        print(f"{'=' * 80}\n")

        # Set new queue file for saving newly discovered links
        self.queue_file = new_queue_file

        # Clear the to_visit set to prepare for crawling
        self.to_visit = set()

        # Track crawled pages count
        pages_crawled = 0

        try:
            # Process each link from the original queue
            for url in original_to_visit:
                print(f"\n{'=' * 80}\nCrawling: {url}\n{'=' * 80}")

                # Skip if already visited
                if url in self.visited:
                    print(f"URL {url} already visited, skipping")
                    continue

                # Mark as visited
                self.visited.add(url)

                # Fetch the page content
                soup = self.fetch_page(url)

                if soup:
                    # Extract the title
                    title = self.get_title(soup)
                    print(f"TITLE: {title}")

                    # Extract the summary
                    summary = self.get_summary(soup)
                    print("\nSUMMARY:")
                    print(summary)

                    # Append to JSON file
                    if self.append_to_json(title, summary, url):
                        print(f"\nAppended to {self.output_file}")
                        pages_crawled += 1
                        print(f"Pages crawled: {pages_crawled}/{len(original_to_visit)}")

                    # Extract links
                    print("\nExtracting links...")
                    new_links = self.extract_links(soup)
                    print(f"Found {len(new_links)} links")

                    # Add new links to the to_visit set (for the new queue)
                    for link in new_links:
                        if link not in self.visited and link not in self.to_visit:
                            self.to_visit.add(link)

                    # Periodically save the new queue (after every page)
                    self.save_queue()

        except KeyboardInterrupt:
            print("\nCrawling interrupted by user")
        finally:
            # Save the new queue before exiting
            self.save_queue()

            print(f"\n{'=' * 80}")
            print(f"Migration process complete.")
            print(f"Processed {pages_crawled} pages from original queue.")
            print(f"Found {len(self.to_visit)} new links saved to {new_queue_file}")

            # Replace the original queue file with the new one
            if os.path.exists(original_queue_file):
                os.remove(original_queue_file)
                print(f"Deleted original queue file: {original_queue_file}")

            os.rename(new_queue_file, original_queue_file)
            print(f"Renamed {new_queue_file} to {original_queue_file}")

            # Update the queue file name back to the original
            self.queue_file = original_queue_file

            print(f"Data saved to {self.output_file}")
            print(f"Queue migration completed successfully.")
            print(f"{'=' * 80}\n")


if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Beshalach"
    crawler = WikipediaCrawler(start_url, max_pages=100, num_threads=8)

    # Use the multi-threaded migration
    crawler.migrate_and_crawl_links_threaded()

    # Alternative using thread pool
    # crawler.migrate_and_crawl_links_with_threadpool()

    # Original methods still available
    # crawler.migrate_and_crawl_links()
    # crawler.crawl()
