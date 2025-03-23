import requests
from bs4 import BeautifulSoup
import re
import json
import os
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string

# Download NLTK resources (uncomment first time)
# nltk.download('punkt')
# nltk.download('stopwords')


class TopicFocusedWikipediaCrawler:
    def __init__(self, start_url, topics, max_pages=50, output_file="wikipedia_data.json", 
                 queue_file="remaining_links.json", num_threads=5,
                 relevance_threshold=0.2, category_depth=2):
        self.start_url = start_url
        self.topics = [topic.lower() for topic in topics]  # List of topics to focus on
        self.relevance_threshold = relevance_threshold  # Minimum relevance score to keep a page
        self.category_depth = category_depth  # How many levels of category pages to follow
        
        self.visited = set()
        self.to_visit = set()
        self.max_pages = max_pages
        self.output_file = output_file
        self.queue_file = queue_file
        self.num_threads = num_threads
        
        # Topic-related sets
        self.relevant_categories = set()
        
        # Locks for thread safety
        self.visited_lock = threading.Lock()
        self.to_visit_lock = threading.Lock()
        self.json_lock = threading.Lock()
        self.category_lock = threading.Lock()
        
        # Stats tracking
        self.pages_crawled = 0
        self.pages_crawled_lock = threading.Lock()
        self.relevant_pages = 0
        self.relevant_pages_lock = threading.Lock()

        # Initialize stopwords for text analysis
        self.stop_words = set(stopwords.words('english'))
        
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
                'User-Agent': 'TopicFocusedWikiCrawler/1.0 (Educational Research Project)'
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

    def get_full_content(self, soup):
        """Extracts more comprehensive content for topic analysis."""
        try:
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return ""

            # Get all paragraphs
            paragraphs = content_div.find_all('p')
            
            # Combine all paragraph text
            full_text = ' '.join([p.text for p in paragraphs])
            
            # Remove citation numbers [1], [2], etc.
            clean_text = re.sub(r'\[\d+\]', '', full_text)
            
            return clean_text.strip()
        except Exception as e:
            print(f"Error extracting full content: {e}")
            return ""

    def get_categories(self, soup):
        """Extract categories from a Wikipedia page."""
        categories = set()
        try:
            # Find the categories box
            category_links = soup.select('div#mw-normal-catlinks ul li a')
            
            for link in category_links:
                category_name = link.text.strip()
                categories.add(category_name.lower())
                
                # Also add the category URL for potential crawling
                href = link.get('href')
                if href and href.startswith('/wiki/Category:'):
                    category_url = f"https://en.wikipedia.org{href}"
                    with self.category_lock:
                        self.relevant_categories.add((category_url, 0))  # 0 is the initial depth
        except Exception as e:
            print(f"Error extracting categories: {e}")
        
        return categories

    def analyze_topic_relevance(self, title, content, categories):
        """
        Analyze if the content is relevant to the specified topics.
        Returns a relevance score between 0.0 and 1.0
        """
        # If no topics specified, consider everything relevant
        if not self.topics:
            return 1.0
            
        # Prepare text for analysis
        title = title.lower()
        content = content.lower()
        
        # Direct checks for exact topic matches in title or categories
        for topic in self.topics:
            # Strong indicators: topic appears in title or categories
            if topic in title:
                return 1.0
                
            for category in categories:
                if topic in category:
                    return 0.9
        
        # Tokenize content and remove stopwords for further analysis
        tokens = word_tokenize(content)
        filtered_tokens = [w for w in tokens 
                          if w.lower() not in self.stop_words
                          and w not in string.punctuation]
        
        # Count word frequencies
        word_counts = Counter(filtered_tokens)
        total_words = len(filtered_tokens)
        
        if total_words == 0:
            return 0.0
            
        # Calculate topic relevance based on frequency of topic-related terms
        topic_relevance = 0.0
        for topic in self.topics:
            # Check for exact topic in filtered tokens
            topic_words = topic.split()
            for word in topic_words:
                if word in word_counts:
                    # Calculate normalized frequency
                    topic_relevance += word_counts[word] / total_words
        
        # Normalize the relevance score
        max_possible_relevance = len(self.topics)  # If all topics appear frequently
        if max_possible_relevance > 0:
            topic_relevance = min(topic_relevance / max_possible_relevance, 1.0)
            
        return topic_relevance

    def extract_links(self, soup, is_category_page=False):
        """Extracts Wikipedia article links from a BeautifulSoup object."""
        links = set()
        
        # Different link extraction logic for category pages
        if is_category_page:
            # Get links from the category members section
            category_members = soup.select('div.mw-category a, div#mw-pages a')
            for link in category_members:
                href = link.get('href')
                if href and href.startswith('/wiki/') and ':' not in href and '#' not in href:
                    full_url = f"https://en.wikipedia.org{href}"
                    links.add(full_url)
                
            # Get subcategory links for deeper crawling if within depth limit
            subcategory_links = soup.select('div#mw-subcategories a')
            for link in subcategory_links:
                href = link.get('href')
                if href and href.startswith('/wiki/Category:'):
                    full_url = f"https://en.wikipedia.org{href}"
                    with self.category_lock:
                        for cat_url, depth in self.relevant_categories:
                            if cat_url == full_url.split('#')[0]:  # Remove fragment identifiers
                                # Already in our list, possibly at a better depth
                                break
                        else:
                            # Extract current depth from source URL
                            current_depth = None
                            for cat_url, depth in self.relevant_categories:
                                if soup.url == cat_url:
                                    current_depth = depth
                                    break
                                    
                            if current_depth is not None and current_depth < self.category_depth - 1:
                                self.relevant_categories.add((full_url, current_depth + 1))
        else:
            # Standard article link extraction
            for link in soup.select('a[href^="/wiki/"]'):
                href = link.get('href')
                if ':' not in href and '#' not in href:  # Filter out non-article pages
                    full_url = f"https://en.wikipedia.org{href}"
                    links.add(full_url)
                
        return links

    def append_to_json(self, title, summary, url, relevance_score, categories):
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
                "wiki_url": url,
                "relevance_score": relevance_score,
                "categories": list(categories)
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

        # Check if it's a category page
        is_category_page = '/wiki/Category:' in url
        
        # Fetch the page content
        soup = self.fetch_page(url)
        if not soup:
            return set()

        # Extract the title
        title = self.get_title(soup)
        print(f"{thread_name} - TITLE: {title}")

        # Extract categories first (needed for relevance calculation)
        categories = self.get_categories(soup)
        
        # Get content for relevance analysis
        full_content = self.get_full_content(soup)
        
        # Calculate topic relevance
        relevance_score = self.analyze_topic_relevance(title, full_content, categories)
        print(f"{thread_name} - Topic relevance: {relevance_score:.2f}")
        
        # Skip pages below relevance threshold unless it's a category page
        # (we still want to crawl categories even if their descriptions aren't relevant)
        if relevance_score < self.relevance_threshold and not is_category_page:
            print(f"{thread_name} - Skipping page (below relevance threshold)")
            return set()
            
        # Extract the summary (only for relevant pages)
        summary = self.get_summary(soup)
        print(f"{thread_name} - Extracted summary")

        # Append to JSON file (only for relevant pages)
        if not is_category_page:  # Don't save category pages to the output file
            if self.append_to_json(title, summary, url, relevance_score, categories):
                with self.pages_crawled_lock:
                    self.pages_crawled += 1
                    current_count = self.pages_crawled
                
                with self.relevant_pages_lock:
                    self.relevant_pages += 1
                    relevant_count = self.relevant_pages
                    
                print(f"{thread_name} - Added to JSON. Relevant pages: {relevant_count}")
        
        # Extract links based on page type
        new_links = self.extract_links(soup, is_category_page)
        print(f"{thread_name} - Found {len(new_links)} links")

        return new_links

    def process_categories(self):
        """Process category pages to find relevant article links."""
        print("\nProcessing categories to find topic-relevant articles...")
        
        # Make a copy to avoid modifying while iterating
        with self.category_lock:
            categories_to_process = list(self.relevant_categories)
            
        for category_url, depth in categories_to_process:
            if depth >= self.category_depth:
                continue
                
            print(f"Processing category: {category_url} (depth {depth})")
            
            # Fetch the category page
            soup = self.fetch_page(category_url)
            if not soup:
                continue
                
            # Extract links from the category page
            new_links = self.extract_links(soup, is_category_page=True)
            print(f"Found {len(new_links)} article links in category")
            
            # Add these links to the to_visit queue
            with self.to_visit_lock:
                for link in new_links:
                    with self.visited_lock:
                        if link not in self.visited:
                            self.to_visit.add(link)
        
        print(f"Finished processing {len(categories_to_process)} categories")

    def migrate_and_crawl_links_threaded(self):
        """
        Multi-threaded version of migrate_and_crawl_links with topic focus
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
        print(f"Starting multi-threaded topic-focused crawling from {original_queue_file}")
        print(f"Topics of interest: {', '.join(self.topics)}")
        print(f"Relevance threshold: {self.relevance_threshold}")
        print(f"Total links to crawl: {len(original_to_visit)}")
        print(f"Using {self.num_threads} threads")
        print(f"{'=' * 80}\n")

        # Set new queue file for saving newly discovered links
        self.queue_file = new_queue_file
        
        # Reset counters
        self.pages_crawled = 0
        self.relevant_pages = 0
        
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
            t = threading.Thread(target=worker, name=f"Worker-{i+1}")
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
                    with self.relevant_pages_lock:
                        relevant_count = self.relevant_pages
                    print(f"\nStatus: {current_count}/{len(original_to_visit)} pages processed")
                    print(f"Found {relevant_count} topic-relevant pages")
                    print(f"Queue size: {work_queue.qsize()}")
                    
                time.sleep(1)
                
            # Wait for all tasks to be processed
            work_queue.join()
            
            # Process category pages to find more relevant content
            self.process_categories()
                
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
            print(f"Crawling process complete.")
            print(f"Processed {self.pages_crawled} pages from original queue.")
            print(f"Found {self.relevant_pages} topic-relevant pages.")
            
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
            print(f"Crawling completed successfully.")
            print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Example: Crawl pages related to quantum physics
    start_url = "https://en.wikipedia.org/wiki/Quantum_mechanics"
    topics = ["quantum physics", "quantum mechanics", "quantum field", "particle physics"]
    
    crawler = TopicFocusedWikipediaCrawler(
        start_url=start_url,
        topics=topics,
        max_pages=100,
        num_threads=6,
        relevance_threshold=0.15,  # Pages must be at least 15% relevant
        category_depth=2  # Follow category pages up to 2 levels deep
    )
    
    # Use the multi-threaded topic-focused crawling
    crawler.migrate_and_crawl_links_threaded()
