import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

class Nutrition(object):
    url = 'https://terms.naver.com/list.naver?cid=59320&categoryId=59320'
    driver_path = 'c:/Program Files/Google/Chrome/chromedriver'
    dict = {}
    df = None
    food_name = []
    food_nut = []

    def scrap_name(self):
        driver = webdriver.Chrome(self.driver_path)
        driver.get(self.url)
        all_div = BeautifulSoup(driver.page_source, 'html.parser')
        ls = all_div.find_all("div", {"class": "subject"})
        for i in ls:
            print(i.find("a").text)
            self.food_name.append(i.find("a").text)
        print(self.food_name)
        driver.close()

    def scrap_nut(self):
        driver = webdriver.Chrome(self.driver_path)
        driver.get(self.url)
        all_div = BeautifulSoup(driver.page_source, 'html.parser')
        ls = all_div.find_all("p", {"class": "desc __ellipsis"})
        print(ls)
        for i in ls:
            print(i.text)

    @staticmethod
    def main():
        nut = Nutrition()
        while 1:
            menu = input('0-Exit\n1-print')
            if menu == '0':
                break
            elif menu == '1':
                nut.scrap_name()
                nut.scrap_nut()
            else:
                continue

Nutrition.main()