from bs4 import BeautifulSoup
import requests
import re


def get_data(url_address):
    request = requests.get(url_address).text
    html_soup = BeautifulSoup(request, 'html5lib')
    return html_soup


url = "https://webpoosh.com/category/%D9%BE%DB%8C%D8%B1%D8%A7%D9%87%D9%86"
soup1 = get_data(url)

for link in soup1.find_all('a'):
    href = link.attrs.get("href")
    try:
        soup = get_data(href)
        with open('/home/vargha/Desktop/image_src2.txt', 'a+') as txt_file:
            for images in soup.find_all('img', {'src': re.compile(r'/([\w_-]+[.](jpg|png|jpeg))$')}):
                src = images['src']
                txt_file.write(src)
                txt_file.write('\n')
    except:
        continue
