import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import urllib.request

# Fill in your credentials + reel URL
USERNAME = "******@gmail.com"
PASSWORD = "******"
REEL_URL = "https://www.instagram.com/reel/DFtrnIOyJ33/"

options = uc.ChromeOptions()
options.add_argument("--headless")  # remove if you want to watch the browser
driver = uc.Chrome(options=options)

try:
    # 1) Log in to Instagram
    driver.get("https://www.instagram.com/accounts/login/")

    # Handle cookie/consent popup (if it appears)
    try:
        accept_cookies_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Accept")]'))
            # Adjust text for your region/language if needed
        )
        accept_cookies_btn.click()
    except:
        pass  # No cookie banner, or different text

    # Wait for the username field
    username_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "username"))
    )
    username_field.send_keys(USERNAME)

    # Wait for and fill password
    password_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "password"))
    )
    password_field.send_keys(PASSWORD)

    # Click the login button
    login_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    login_button.click()
    time.sleep(5)  # Wait for the login to complete

    # 2) Go to the Reel (once logged in)
    driver.get(REEL_URL)
    time.sleep(8)  # Let reel content load

    # 3) Grab the real video URL
    #    a) Check if there's a <video> tag with an .mp4 (best-case scenario)
    try:
        video_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "video"))
        )
        # This might be "blob:" if Instagram loads it that way,
        # but let's see if we get a direct .mp4
        src_url = video_element.get_attribute("src")
    except:
        src_url = None

    if src_url and src_url.startswith("http"):
        # We found a direct .mp4-like URL
        print("Real video URL:", src_url)
        urllib.request.urlretrieve(src_url, "reel_video.mp4")
        print("Video downloaded successfully.")
    else:
        # 4) If we got a 'blob:' or None, we can try the performance log approach
        print("We got a blob: or no URL. Let's capture network traffic for .mp4")

        # Let's re-initialize a new driver with devtools to capture requests
        driver.quit()
        options = uc.ChromeOptions()
        options.add_argument("--headless")
        driver = uc.Chrome(options=options, enable_cdp_events=True)

        # Re-log in quickly
        driver.get("https://www.instagram.com/accounts/login/")
        time.sleep(3)

        # Repeat login steps (shortened, no cookie handling for brevity)
        driver.find_element(By.NAME, "username").send_keys(USERNAME)
        driver.find_element(By.NAME, "password").send_keys(PASSWORD)
        driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]').click()
        time.sleep(5)

        # Capture .mp4 from network requests
        mp4_urls = []


        def on_response_received(**kwargs):
            try:
                resp_url = kwargs["response"]["url"]
                if ".mp4" in resp_url:
                    mp4_urls.append(resp_url)
            except:
                pass


        driver.add_cdp_listener("Network.responseReceived", on_response_received)

        driver.get(REEL_URL)
        time.sleep(10)  # Let the reel load and the requests happen

        if mp4_urls:
            real_mp4 = mp4_urls[0]
            print("Real video URL from logs:", real_mp4)
            urllib.request.urlretrieve(real_mp4, "reel_video.mp4")
            print("Video downloaded from logs successfully.")
        else:
            print("Could not capture .mp4 from network logs either.")

finally:
    driver.quit()
