from bs4 import BeautifulSoup
import requests
import re
import time
from selenium import webdriver
import csv

driver = webdriver.PhantomJS("/home/vargha/phantomjs-2.1.1-linux-x86_64/bin/phantomjs")


def get_data(url_address):
    request = requests.get(url_address).text
    html_soup = BeautifulSoup(request, 'lxml')
    return html_soup


base_url = "https://www.modiseh.com"
url = "https://www.modiseh.com/men/apparel/shirts?modelattribute=%D8%A2%D8%B3%D8%AA%DB%8C%D9%86-%D8%A8%D9%84%D9%86%D8%AF"
name_list = []

browser = webdriver.Firefox()
browser.get(url)
len_of_page = browser.execute_script(
    "window.scrollTo(0, document.body.scrollHeight);var len_of_page=document.body.scrollHeight;return len_of_page;")
match = False
while not match:
    last_count = len_of_page
    time.sleep(10)
    len_of_page = browser.execute_script(
        "window.scrollTo(0, document.body.scrollHeight);var len_of_page=document.body.scrollHeight;return len_of_page;")
    if last_count == len_of_page:
        match = True

source_data = browser.page_source
soup1 = BeautifulSoup(source_data, features='lxml')
header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image']

with open('/home/vargha/Desktop/image_src3.csv', 'a+', encoding='UTF8') as txt_file:
    writer = csv.writer(txt_file)
    writer.writerow(header)
    for link in soup1.find_all('a', {'class': "product-url"}):
        href_list = []
        href = link.attrs.get("href")
        try:
            soup = get_data(base_url + href)
            for file in soup.find_all('figcaption', {'id': 'thumbnail'}):
                images = file.find('img', {'src': re.compile(r'/([\w_-]+[.](jpg|png|jpeg))$')})
                src = images['src']
                name = images['title']
                clothes_filters = ['تیشرت', 'شلوار', 'پیراهن']
                if name not in name_list:
                    name_list.append(name)
                    if any(clothes in name for clothes in clothes_filters):
                        txt_file.write(name)
                        txt_file.write(', ')
                        txt_file.write(base_url + href)
                        txt_file.write(', ')
                        element = file.img
                        parent_tag = element.parent
                        nextSiblings = parent_tag.find_all("img")
                        for nextSibling in nextSiblings:
                            src_image = nextSibling['data-img']
                            txt_file.write(base_url + src_image)
                            txt_file.write(', ')
                        txt_file.write('\n')
        except:
            continue
