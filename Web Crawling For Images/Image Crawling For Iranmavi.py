from bs4 import BeautifulSoup
import requests
import re
import csv


def get_data(url_address):
    request = requests.get(url_address).text
    html_soup = BeautifulSoup(request, 'lxml')
    return html_soup


def write_csv(soup2):
    for link in soup2.find_all('div', {'class': "item-area"}):
        for links in link.find_all('a', {'class': 'product-image'}):
            href = links.attrs.get("href")
            if href not in href_list:
                href_list.append(href)
                try:
                    soup = get_data(href)
                    for file in soup.find_all('div', {'class': 'MagicToolboxSelectorsContainer'}):
                        images = file.find('a', {'href': re.compile(r'/([\w_-]+[.](jpg|png|jpeg))$')})
                        name = images['title']
                        clothes_filters = ['تی شرت', 'شلوار', 'پیراهن']
                        if any(clothes in name for clothes in clothes_filters):
                            txt_file.write(name)
                            txt_file.write(', ')
                            txt_file.write(href)
                            txt_file.write(', ')
                            element = file.a
                            parent_tag = element.parent
                            next_siblings = parent_tag.find_all("a")
                            for next_sibling in next_siblings:
                                src_image = next_sibling['href']
                                txt_file.write(src_image)
                                txt_file.write(', ')
                            txt_file.write('\n')
                except:
                    continue


flag = False
href_list = []
url = "https://www.iranmavi.com/men/t-shirt.html?p=1"
soup1 = get_data(url)
header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Ninth Image', 'Tenth Image', 'Price', 'Value of Discount',
          'Collar', 'Sleeve']
with open('/home/vargha/Desktop/image_src3.csv', 'w', encoding='UTF8') as txt_file:
    writer = csv.writer(txt_file)
    writer.writerow(header)
    for pages in soup1.find_all('div', {'class': "pager"}):
        current_page = pages.find('li', {'class': 'current'})
        if not flag:
            soup2 = soup1
            write_csv(soup2)
            flag = True
        else:
            for hrefs in pages.find_all('a'):
                page = hrefs.attrs.get("href")
                soup2 = get_data(page)
                write_csv(soup2)
