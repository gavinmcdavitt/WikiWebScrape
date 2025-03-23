import requests
from bs4 import BeautifulSoup
import re
import json
import os


class WikipediaCrawler:
    def __init__(self, start_url, max_pages=50, output_file="wikipedia_data.json", queue_file="remaining_links.json"):
        self.start_url = start_url
        self.visited = set()
        self.to_visit = set()
        self.max_pages = max_pages
        self.output_file = output_file
        self.queue_file = queue_file

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
        with open(self.queue_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.to_visit), f, indent=2)
        print(f"Saved {len(self.to_visit)} links to queue file")

    def fetch_page(self, url):
        """Fetches a Wikipedia page and returns its parsed soup."""
        try:
            response = requests.get(url)
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

    def crawl(self):
        """Crawls Wikipedia pages recursively using a stack."""
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
        """
        Migrate links from the original queue file to a new queue file while crawling.
        Process:
        1. Create a new queue file (remaining_links_2.json)
        2. Crawl all links from the original queue (remaining_links.json)
        3. Store newly discovered links in the new queue file
        4. Replace the original queue file with the new one when done
        """
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
    start_url = "https://en.wikipedia.org/wiki/Assassination_of_John_F._Kennedy"
    crawler = WikipediaCrawler(start_url, max_pages=100)
    
    # To use the new migration functionality, uncomment this line:
     crawler.migrate_and_crawl_links()
    
    # Or use the original crawl function:
    crawler.crawl()
