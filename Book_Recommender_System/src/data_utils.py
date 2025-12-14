import pandas as pd
import os
import time
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus

RAW_PATH = "D:\\PYTHON\\Project\\Book_Recommender_System\\data\\raw\\global_book_insights_10k.csv"
PROCESSED_PATH = "D:\\PYTHON\\Project\\Book_Recommender_System\\data\\processed\\cleaned_books.csv"
SAVE_EVERY = 30
MAX_WORKERS = 10


def clean_dataset(df):
    """Apply all cleaning steps to the dataset"""
    if df.columns[0] in ['Unnamed: 0', 'index']:
        df = df.iloc[:, 1:]
    
    df["Description"] = df["Description"].fillna("")
    df["Book"] = df["Book"].astype(str).str.strip()
    df["Author"] = df["Author"].astype(str).str.strip()
    df["Genres"] = df["Genres"].astype(str)
    df = df.drop_duplicates(subset=["Book", "Author"]).reset_index(drop=True)
    df = df.drop(columns=['cover_source'])
    df['Cover_URL'] = df['cover_url']
    df = df.drop(columns=['cover_url'])
    
    return df


def scrape_goodreads_cover(url, retries=2):
    """Scrape cover from Goodreads with multiple fallback methods"""
    if not url or pd.isna(url):
        return None
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.goodreads.com/',
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=8)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                img = soup.find('img', class_='ResponsiveImage')
                if img and img.get('src'):
                    return img['src']
                
                meta = soup.find('meta', property='og:image')
                if meta and meta.get('content'):
                    content = meta['content']
                    if 'nophoto' not in content.lower():
                        return content
                
                cover_div = soup.find('div', class_='BookCover')
                if cover_div:
                    img = cover_div.find('img')
                    if img and img.get('src'):
                        return img['src']
                
                patterns = [
                    r'https://images\.gr-assets\.com/books/[^"\'>\s]+',
                    r'https://i\.gr-assets\.com/images/[^"\'>\s]+',
                    r'https://m\.media-amazon\.com/images/[^"\'>\s]+'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, response.text)
                    if match:
                        url_found = match.group(0)
                        if 'nophoto' not in url_found.lower():
                            return url_found
            
            elif response.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
                
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
    
    return None


def get_google_books_cover(title, author):
    """Fetch cover from Google Books API"""
    try:
        query = f'intitle:"{title}" inauthor:"{author}"'
        query_encoded = quote_plus(query)
        url = f"https://www.googleapis.com/books/v1/volumes?q={query_encoded}&maxResults=3"
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('totalItems', 0) > 0:
                for item in data.get('items', []):
                    volume_info = item.get('volumeInfo', {})
                    
                    api_title = volume_info.get('title', '').lower()
                    if title.lower()[:20] in api_title or api_title[:20] in title.lower():
                        
                        image_links = volume_info.get('imageLinks', {})
                        for size in ['large', 'medium', 'thumbnail', 'smallThumbnail']:
                            if size in image_links:
                                cover_url = image_links[size]
                                cover_url = cover_url.replace('zoom=1', 'zoom=2')
                                cover_url = cover_url.replace('&edge=curl', '')
                                return cover_url
    
    except Exception as e:
        pass
    
    return None


def get_openlibrary_cover(title, author):
    """Fetch cover from Open Library API"""
    try:
        query = f"{title} {author}".strip()
        query_encoded = quote_plus(query)
        url = f"https://openlibrary.org/search.json?q={query_encoded}&limit=3"
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            docs = data.get('docs', [])
            
            for doc in docs:
                api_title = doc.get('title', '').lower()
                if title.lower()[:15] in api_title or api_title[:15] in title.lower():
                    
                    cover_id = doc.get('cover_i')
                    if cover_id:
                        return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    
    except Exception as e:
        pass
    
    return None


def fetch_cover_hybrid(row_data):
    """Fetch cover using all 3 methods with priority: Goodreads > Google Books > Open Library"""
    idx, book, author, url = row_data
    
    if url and pd.notna(url):
        cover = scrape_goodreads_cover(url)
        if cover and 'nophoto' not in cover.lower():
            return idx, cover, "Goodreads"
    
    time.sleep(0.05)
    
    cover = get_google_books_cover(book, author)
    if cover:
        return idx, cover, "Google Books"
    
    cover = get_openlibrary_cover(book, author)
    if cover:
        return idx, cover, "Open Library"
    
    return idx, None, "Failed"


def process_dataset():
    """Main function to clean and enrich dataset with covers"""
    
    print("=" * 60)
    print("📚 NOVA BOOKS - Advanced Dataset Processor")
    print("=" * 60)
    
    print("\n[1/5] Loading raw dataset...")
    df_raw = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df_raw)} books")
    
    print("\n[2/5] Cleaning dataset...")
    before_dedup = len(df_raw)
    df_raw = clean_dataset(df_raw)
    after_dedup = len(df_raw)
    print(f"Removed {before_dedup - after_dedup} duplicate books")
    print(f"Clean dataset: {after_dedup} unique books")
    
    print("\n[3/5] Checking for existing progress...")
    if os.path.exists(PROCESSED_PATH):
        df_processed = pd.read_csv(PROCESSED_PATH)
        print(f"Found existing file with {len(df_processed)} books")
        
        df = pd.merge(
            df_raw, 
            df_processed[["Book", "Author", "cover_url", "cover_source"]],
            on=["Book", "Author"], 
            how="left"
        )
    else:
        df = df_raw.copy()
        df["cover_url"] = ""
        df["cover_source"] = ""
        print("No existing file found, starting fresh")
    
    print("\n[4/5] Fetching book covers...")
    df["cover_url"] = df["cover_url"].fillna("")
    needs_cover = df[df["cover_url"] == ""].copy()
    total_to_process = len(needs_cover)
    
    if total_to_process == 0:
        print("All books already have covers")
    else:
        print(f"{total_to_process} books need covers")
        print(f"Using {MAX_WORKERS} parallel workers")
        print(f"Saving progress every {SAVE_EVERY} rows")
        print("-" * 60)
        
        start_time = time.time()
        processed_count = 0
        success_count = 0
        
        tasks = [
            (idx, row["Book"], row["Author"], row.get("URL", ""))
            for idx, row in needs_cover.iterrows()
        ]
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {
                executor.submit(fetch_cover_hybrid, task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                try:
                    idx, cover_url, source = future.result()
                    
                    if cover_url:
                        df.at[idx, "cover_url"] = cover_url
                        df.at[idx, "cover_source"] = source
                        success_count += 1
                        status = "SUCCESS"
                    else:
                        df.at[idx, "cover_url"] = ""
                        df.at[idx, "cover_source"] = "Failed"
                        status = "FAILED"
                    
                    processed_count += 1
                    
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    eta = (total_to_process - processed_count) / rate if rate > 0 else 0
                    
                    book_name = df.loc[idx, "Book"][:40]
                    print(f"[{processed_count}/{total_to_process}] "
                          f"{status:7} | {book_name:40} | {source:15} | "
                          f"Success Rate: {success_count}/{processed_count} "
                          f"({success_count/processed_count*100:.1f}%) | "
                          f"ETA: {eta/60:.1f}m")
                    
                    if processed_count % SAVE_EVERY == 0:
                        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
                        df.to_csv(PROCESSED_PATH, index=False)
                        print(f"Progress saved at {processed_count}/{total_to_process}")
                
                except Exception as e:
                    print(f"Error processing task: {e}")
    
    print("\n[5/5] Finalizing...")
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    
    total_books = len(df)
    books_with_covers = len(df[df["Cover_URL"] != ""])
    coverage_pct = (books_with_covers / total_books * 100) if total_books > 0 else 0
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total Books: {total_books}")
    print(f"Books with Covers: {books_with_covers} ({coverage_pct:.1f}%)")
    print(f"Books without Covers: {total_books - books_with_covers}")
    print(f"\nSaved to: {PROCESSED_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        process_dataset()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("Progress has been saved. You can re-run to continue.")