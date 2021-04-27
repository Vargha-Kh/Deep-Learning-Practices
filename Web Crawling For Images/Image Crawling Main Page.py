from bs4 import BeautifulSoup
import requests
import re


def get_data(url_address):
    request = requests.get(url_address).text
    html_soup = BeautifulSoup(request, 'html5lib')
    return html_soup


# For Only Main Page
url = "https://corumofficial.com/product-category/men/category-men-collection/t-shirt/men-s-tshirt/"
soup = get_data(url)
with open('/home/vargha/Desktop/image_src2.txt', 'a+') as txt_file:
    for file in soup.find_all('div', {'class': 'product-small box'}):
        for images in file.find_all('img', {'src': re.compile(r'/([\w_-]+[.](jpg|png|jpeg))$')}):
            src = images['src']
            txt_file.write(src)
            txt_file.write('\n')
