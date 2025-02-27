import os
import requests
from bs4 import BeautifulSoup

# Base URL of the public access link

# Create an output directory in the current working directory
output_dir = os.path.join(os.getcwd(), "directives")
os.makedirs(output_dir, exist_ok=True)

def download_pdfs(series_str):
    base_url = f"https://www.weather.gov/directives/{series_str}"
    try:
        # Fetch the webpage
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the webpage content
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all links to PDF files
        pdf_links = soup.find_all("a", href=lambda href: href and href.lower().endswith(".pdf"))

        if not pdf_links:
            print("No PDFs found on the page.")
            return

        print(f"Found {len(pdf_links)} PDFs. Starting download...")

        # Download each PDF
        for link in pdf_links:
            pdf_url = link["href"]

            # Ensure the link is absolute
            if not pdf_url.startswith("http"):
                pdf_url = requests.compat.urljoin(base_url, pdf_url)

            # Check if the URL actually points to a PDF
            head_response = requests.head(pdf_url, allow_redirects=True)
            if head_response.headers.get("Content-Type") != "application/pdf":
                print(f"Skipping non-PDF link: {pdf_url}")
                continue

            # Get the PDF filename
            pdf_filename = os.path.join(output_dir, os.path.basename(pdf_url))

            # Download and save the PDF
            with requests.get(pdf_url, stream=True) as pdf_response:
                pdf_response.raise_for_status()
                with open(pdf_filename, "wb") as pdf_file:
                    for chunk in pdf_response.iter_content(chunk_size=1024):
                        pdf_file.write(chunk)

            print(f"Downloaded: {pdf_filename}")

        print("All PDFs have been downloaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
  series_list = ["001", "010", "020", "030", "040", "050", "060", "070", "090", "100"]
  for series_str in series_list:
    download_pdfs(series_str)