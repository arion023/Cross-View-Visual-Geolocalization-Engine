from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select

from enum import Enum
from time import sleep
import os

from PIL import Image
import pandas as pd
import shutil


# -----------------------------------#
#                                   #
#   OPTIONS TO DEFINE BELOW ENUMS   #
#                                   #
# -----------------------------------#

# TURN OFF DISPLAY UP-SCALING (usually default behaviour for displayes with higher reolutions)


# mapping to the id of the option in html selector object
class ZOOM(Enum):
    ZOOM12 = 1  # 1:150 000
    ZOOM13 = 2  # 1:70 000
    ZOOM14 = 3  # 1:35 000
    ZOOM15 = 4  # 1:15 000
    ZOOM16 = 5  # 1:8000
    ZOOM17 = 6  # 1:4000
    ZOOM18 = 7  # 1:2000
    ZOOM19 = 8  # 1:1000
    ZOOM20 = 9  # 1:500
    ZOOM21 = 10  # 1:250


class ORIENTATION(Enum):
    NORTH = "buttonN"
    SOUTH = "buttonS"
    WEST = "buttonW"
    EAST = "buttonE"


# mapping for id numbers of html selector objects
class LAYER_TYPE(Enum):
    ORTO = 0
    OBLIQUE = 1  # from 45 angle
    LIDAR = 2
    OPENSTREET = 3


# ----------------- OPTIONS ------------------#

PATH_TO_MOZILLA = "C:/Program Files/Mozilla Firefox/firefox.exe"

URL = "https://ukosne.um.warszawa.pl/"

FILE_WITH_LOCATIONS = "D:/datasets/Warsaw/locations/basic.csv"

ZOOMS = [ZOOM.ZOOM20]# [ZOOM.ZOOM18, ZOOM.ZOOM19, ZOOM.ZOOM20, ZOOM.ZOOM21]

TYPE_OF_LAYER = LAYER_TYPE.OBLIQUE

VIEW_ORIENTATION = ORIENTATION.EAST

PATH_TO_SAVE = f"D:/datasets/Warsaw/sat/waw/2022/oblique/east/"

DEFAULT_DOWNLOAD_PATH = "C:/Users/masik/Downloads/"
DEFAULT_DOWNLOAD_FILE_NAME = "export_Obliview.jpg"

# --------------------------------------------#


def close_cookies(driver):
    refuse_cookies_button = driver.find_element(By.CLASS_NAME, "reg-popup-closebutton")
    refuse_cookies_button.click()


def select_layer_type(driver, layer=LAYER_TYPE.ORTO):
    show_selector_options = driver.find_element(By.ID, "LayerButtonTop1")

    show_selector_options.click()  # open

    # in case clicking doesnt work you can hover on that element
    # ActionChains(driver).move_to_element(show_selector_options).perform()

    selectors = driver.find_elements(By.CLASS_NAME, "baseLayerPicker-item")

    selectors[layer.value].click()

    show_selector_options.click()  # close
    sleep(0.1)


def select_orientation(driver, orientation=ORIENTATION.NORTH):
    orienation_button = driver.find_element(By.ID, orientation.value)
    orienation_button.click()


def select_zoom(driver, zoom_level=ZOOM.ZOOM12):
    show_zoom_options = driver.find_element(By.ID, "optionsButton")
    show_zoom_options.click()  # open

    zoom_options = Select(driver.find_element(By.ID, "setScale"))
    zoom_options.select_by_index(zoom_level.value)
    sleep(0.1)
    show_zoom_options.click()  # close


def search_coordinates(driver, coords="52.156944,21.080122"):
    # show search field (make it active, without that there is no possibility to search)
    search_show = driver.find_element(By.ID, "searchIcon")
    search_show.click()  # open

    # show search type options
    selector_show_button = driver.find_element(By.ID, "searchOptionLang")
    selector_show_button.click()

    # select number 2 select option (coordinates)
    select_coordinates = driver.find_element(By.ID, "searchOption#2")
    select_coordinates.click()

    # insert input and search
    search_input = driver.find_element(By.ID, "search-input")
    search_input.send_keys(coords)
    search_input.send_keys(Keys.ENTER)

    search_clear = driver.find_element(By.ID, "buttonSearchClear")
    search_clear.click()

    search_show.click()  # close


def dump_view(driver):
    sleep(3)
    export_show = driver.find_element(By.ID, "exportImportTitle")
    export_show.click()  # open

    export_tile = driver.find_element(By.ID, "exportTitle")
    export_tile.click()

    export_button = driver.find_element(By.ID, "buttonLoadExport")
    export_button.click()

    close_button = driver.find_element(By.ID, "closeButton")
    close_button.click()  # close export pop up window
    sleep(0.5)


def run():
    # in mozzila explorer frame and web options widith and height
    mozzila_width = 12
    mozzila_height = 91

    # service trademark height (from bottom)
    trademark_height = 60

    # if file locations is default, u can skip this
    options = Options()
    options.binary_location = PATH_TO_MOZILLA
    driver = webdriver.Firefox(options=options)

    driver.set_window_size(1200 + mozzila_width, 1260 + mozzila_height)

    driver.get(URL)
    sleep(1)
    try:
        close_cookies(driver)
    except Exception as e:
        print("WARNING. There is no cookies to close.")
        print(e)
    select_layer_type(driver, TYPE_OF_LAYER)
    select_orientation(driver, VIEW_ORIENTATION)
    for z in ZOOMS:
        select_zoom(driver, z)
        locations = pd.read_csv(FILE_WITH_LOCATIONS)

        for i, lat, lon in zip(
            locations["place id"], locations["latitude"], locations["longitude"]
        ):
            # if i in range(54,65):
            search_coordinates(driver, coords=str(lat) + "," + str(lon))
            dump_view(driver)
            destiantion = PATH_TO_SAVE + f"{z.name[-2:]}/{i:03}.jpg"
            os.makedirs(os.path.dirname(destiantion), exist_ok=True)
            shutil.move(
                DEFAULT_DOWNLOAD_PATH + DEFAULT_DOWNLOAD_FILE_NAME, destiantion
            )
            image = Image.open(destiantion)
            width, height = image.size
            image = image.crop((0, 0, width, height - trademark_height))
            image.save(destiantion)
    driver.close()


if __name__ == "__main__":
    run()
