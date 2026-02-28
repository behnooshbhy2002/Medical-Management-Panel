"""
DrugBank Biotech Drugs Crawler (without RabbitMQ alternative version)
- Uses Selenium with existing Chrome driver
- Saves scraped data to a JSON file
- Iterates through all pages using the 'next' button
"""

import json
import time
import math
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import pika

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CHROME_DRIVER_PATH = r"D:\Desktop\chromedriver-win64\chromedriver.exe"   # Put your chromedriver path here

OUTPUT_FILE        = "drugs_output.json"

RABBITMQ_URL = "amqp://guest:guest@localhost:30672/"
QUEUE_NAME   = "drugbank_biotech_drugs"

START_URL = "https://go.drugbank.com/drugs?approved=1&c=name&d=up"


PAGE_LOAD_WAIT = 10   # seconds to wait for page load
BETWEEN_PAGES  = 2    # delay between pages (be nice to the server)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Selenium setup
# ─────────────────────────────────────────────

def build_driver() -> webdriver.Chrome:
    options = Options()
    # options.add_argument("--headless")          # uncomment for headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    service = Service(executable_path=CHROME_DRIVER_PATH)
    driver  = webdriver.Chrome(service=service, options=options)
    log.info("✅ Chrome driver started")
    return driver


# ─────────────────────────────────────────────
# Parse table rows
# ─────────────────────────────────────────────

def parse_rows(driver) -> list:
    rows = []
    try:
        tbody = driver.find_element(By.CSS_SELECTOR, "table#drugs-table tbody")
        trs   = tbody.find_elements(By.TAG_NAME, "tr")
    except NoSuchElementException:
        log.warning("Drugs table not found!")
        return rows

    for tr in trs:
        try:
            name = tr.find_element(By.CSS_SELECTOR, "td.name-value").text.strip()
        except NoSuchElementException:
            name = ""

        try:
            weight = tr.find_element(By.CSS_SELECTOR, "td.weight-value").text.strip()
        except NoSuchElementException:
            weight = ""

        try:
            description = tr.find_element(By.CSS_SELECTOR, "td.description-value").text.strip()
        except NoSuchElementException:
            description = ""

        try:
            categories_el = tr.find_elements(By.CSS_SELECTOR, "td.categories-value a")
            categories = [a.text.strip() for a in categories_el]
        except NoSuchElementException:
            categories = []

        try:
            link = tr.find_element(By.CSS_SELECTOR, "td.name-value a").get_attribute("href")
        except NoSuchElementException:
            link = ""

        if name:
            rows.append({
                "name":        name,
                "weight":      weight,
                "description": description,
                "categories":  categories,
                "link":        link,
            })

    return rows


# ─────────────────────────────────────────────
# Get total number of pages
# ─────────────────────────────────────────────

def get_total_pages(driver) -> int:
    try:
        info_text     = driver.find_element(
            By.XPATH, "//*[contains(text(),'Displaying drugs')]"
        ).text
        total_records = int(info_text.split("of")[1].split("in")[0].strip().replace(",", ""))
        first_record  = int(info_text.split("Displaying drugs")[1].split("-")[0].strip())
        last_record   = int(info_text.split("-")[1].split("of")[0].strip())
        per_page      = last_record - first_record + 1
        total_pages   = math.ceil(total_records / per_page)
        log.info(f"📄 Total records: {total_records} | per page: {per_page} | total pages: {total_pages}")
        return total_pages
    except Exception as e:
        log.warning(f"Could not determine total pages: {e}")
        try:
            page_links = driver.find_elements(By.CSS_SELECTOR, "ul.pagination li.page-item a.page-link")
            numbers = [int(a.text.strip()) for a in page_links if a.text.strip().isdigit()]
            if numbers:
                return max(numbers)
        except Exception:
            pass
        return 1


# ─────────────────────────────────────────────
# Click Next button
# ─────────────────────────────────────────────

def click_next(driver) -> bool:
    try:
        next_li = driver.find_element(By.CSS_SELECTOR, "li.page-item.next")
        if "disabled" in next_li.get_attribute("class"):
            log.info("Next button is disabled — last page reached.")
            return False
        next_a = next_li.find_element(By.CSS_SELECTOR, "a.page-link")
        driver.execute_script("arguments[0].click();", next_a)
        return True
    except NoSuchElementException:
        log.info("Next button not found.")
        return False


def connect_rabbitmq():
    while True:
        try:
            log.info("🐰 Trying to connect to RabbitMQ...")
            connection = pika.BlockingConnection(
                pika.URLParameters(RABBITMQ_URL)
            )
            log.info("✅ Connected to RabbitMQ.")
            return connection
        except Exception as e:
            log.warning(f"⏳ RabbitMQ not ready, retrying in 5 seconds... {e}")
            time.sleep(5)


# ─────────────────────────────────────────────
# Main crawling loop
# ─────────────────────────────────────────────

def crawl():
    driver = build_driver()
    connection = None
    channel = None
    all_drugs = []

    try:
        log.info(f"🌐 Opening start URL: {START_URL}")
        driver.get(START_URL)

        wait = WebDriverWait(driver, PAGE_LOAD_WAIT)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table#drugs-table")))

        # ── Connect to RabbitMQ ──
        log.info("🐰 Connecting to RabbitMQ...")
        connection = connect_rabbitmq()
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        log.info(f"✅ Queue '{QUEUE_NAME}' declared.")

        total_pages = get_total_pages(driver)
        current_page = 1

        while True:
            log.info(f"📃 Page {current_page} of {total_pages}")

            drugs = parse_rows(driver)
            for drug in drugs:
                drug["page"] = current_page
                
                # ── Publish to RabbitMQ ──
                try:
                    message = json.dumps(drug, ensure_ascii=False).encode("utf-8")
                    channel.basic_publish(
                        exchange="",
                        routing_key=QUEUE_NAME,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2),  # persistent
                    )
                except Exception as e:
                    log.warning(f"⚠️ Error publishing to queue: {e}")

            all_drugs.extend(drugs)
            log.info(f"   → Found {len(drugs)} drugs (total: {len(all_drugs)})")

            # Save progress to JSON
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_drugs, f, ensure_ascii=False, indent=2)

            if current_page >= total_pages or not click_next(driver):
                break

            current_page += 1
            time.sleep(BETWEEN_PAGES)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table#drugs-table tbody tr")))

    finally:
        driver.quit()
        if connection is not None:
            try:
                connection.close()
                log.info("🐰 RabbitMQ connection closed.")
            except:
                pass
        log.info(f"🏁 Crawling finished. {len(all_drugs)} drugs saved to JSON + sent to RabbitMQ.")


if __name__ == "__main__":
    crawl()