import csv
import time

from parsel import Selector
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


def scrape_google_jobs(selector, query_string):
    google_jobs_results = []

    # Select title, company and location. Select description and other info
    for result, result_main in zip(selector.css(".iFjolb"), selector.css(".pE8vnd")):

        # Place selected data from the css into results
        title = result.css(".BjJfJf::text").get()
        company = result.css(".vNEEBe::text").get()

        # Job location and source
        container = result.css(".Qk80Jf::text").getall()
        location = container[0]
        via = container[1]

        # Addtional info e.g. decriptions
        extensions = result.css(".KKh3md span::text").getall()
        description = result_main.css(".YgLbBe span::text").getall()
        link = result_main.css(".EDblX a::attr(href)").getall()

        google_jobs_results.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "via": via,
                "extensions": extensions,
                "link": link,
                "description": description,
            }
        )
    # Specify the file path where you want to save the CSV file
    job_title = query_string.replace(" ", "_").lower()
    csv_file_path = f"data/scraped_job_descriptions_{job_title}.csv"

    # Extract the keys from the first dictionary to use as the header row
    header = google_jobs_results[0].keys()

    # Open the CSV file in write mode and create a CSV writer
    with open(csv_file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)

        # Write the header row
        writer.writeheader()

        # Write each dictionary as a row in the CSV file
        writer.writerows(google_jobs_results)

    # Print a message to confirm the CSV file was saved
    print("Scraped Job Descriptions saved to CSV file:", csv_file_path)
    

def selenium_scrape(query_string="Data Scientist"):
    params = {
        "q": query_string,  # search string
        "ibp": "htl;jobs",  # google jobs
        "uule": "w+CAIQICIJc2luZ2Fwb3Jl",  # encoded location (SG)
        "hl": "en",  # language
        "gl": "sg",  # country of the search
        "date_posted": "3days",
    }

    URL = f"https://www.google.com/search?q={params['q']}&ibp={params['ibp']}&uule={params['uule']}&hl={params['hl']}&gl={params['gl']}&date_posted={params['date_posted']}"
