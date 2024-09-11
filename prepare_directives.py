import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import time


def get_pdf_links(url):
   try:
       response = requests.get(url)
       response.raise_for_status()
       soup = BeautifulSoup(response.content, 'html.parser')
       pdf_links = []
      
       # Find all list items
       list_items = soup.find_all('li')
      
       for item in list_items:
           # Check if the item's text ends with '.pdf'
           if item.text.strip().lower().endswith('.pdf'):
               pdf_links.append(item.text.strip())
      
       print(f"Found {len(pdf_links)} PDF links on the page.")
       return pdf_links
   except requests.exceptions.RequestException as e:
       print(f"Error fetching the webpage: {e}")
       return []


def download_pdfs(base_url):
   pdf_links = get_pdf_links(base_url)
  
   if not pdf_links:
       print("No PDF links found on the page.")
       return
  
   # Create 'directives' directory if it doesn't exist
   directives_dir = 'directives'
   os.makedirs(directives_dir, exist_ok=True)


   # Construct the base URL for PDF files
   pdf_base_url = "https://www.weather.gov/media/directives/080_pdfs_archived/"


   for index, filename in enumerate(pdf_links, start=1):
       file_url = urljoin(pdf_base_url, filename)


       try:
           print(f"Attempting to download {filename} ({index}/{len(pdf_links)})")
           response = requests.get(file_url)
           response.raise_for_status()


           file_path = os.path.join(directives_dir, filename)
           with open(file_path, 'wb') as file:
               file.write(response.content)
          
           print(f"Successfully downloaded: {filename}")
          
           # Add a small delay between downloads
           time.sleep(1)


       except requests.exceptions.RequestException as e:
           print(f"Error downloading {filename}: {e}")


if __name__ == "__main__":
   base_url = "https://www.weather.gov/directives/080-archives-list"
   print(f"Starting download process from {base_url}")
   download_pdfs(base_url)
   print("Download process completed.")
   print(f"Files should be saved in: {os.getcwd()}/directives")
