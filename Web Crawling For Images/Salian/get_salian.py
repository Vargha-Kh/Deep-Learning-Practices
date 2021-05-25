import time
from bs4 import BeautifulSoup
import requests
import pandas as pd
from unidecode import unidecode
import re

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Price', 'Discounted Price',
          'Collar']

def get_salian():
    categories = {
        # "pants": "https://www.salian.com/men/clothes/pants.html?p=",
        "ladies_pants": "https://www.salian.com/women/clothes/pants.html?p=",
        "shirt": "https://www.salian.com/men/clothes/casual-shirt.html?p=",
        "dress-shirt": "https://www.salian.com/men/clothes/dress-shirt.html?p=",
        "t-shirt": "https://www.salian.com/men/clothes/t-shirt.html?p="}
    base_link = "https://www.salian.com"
    for category_name, category in categories.items():
        df_lst = []
        href_list = []
        for p in range(1, 5):
            file_link = category + str(p)
            page = requests.get(file_link)
            soup = BeautifulSoup(page.content, 'html.parser')
            a_links = soup.find_all("a", class_="product-image")
            if not a_links:
                break
            for a in a_links:
                href = a['href']
                title = a['title']
                if href not in href_list:
                    href_list.append(href)
                    clothes_filters = ['تی شرت', 'شلوار', 'پیراهن', 'تیشرت', 'PANT', 'T-SHIRT', 'SHIRT']
                    if any(clothes in title for clothes in clothes_filters):
                        row = [title, href]
                        page_ = requests.get(href)
                        soup_ = BeautifulSoup(page_.content, 'html.parser')
                        divs = soup_.find(id='amasty_gallery')
                        if len(divs) == 0:
                            continue
                        else:
                            a_inner_links = divs.find_all('a')
                        for img in a_inner_links:
                            img_link = img['data-zoom-image']
                            row.append(img_link)
                        print(len(row) - 1, " images has been added!")
                        number_of_pictures = 8 - (len(row) - 2)
                        if number_of_pictures != 0:
                            for i in range(number_of_pictures):
                                row.append([])

                        price = soup_.find_all('div', {'class': 'price-box'})
                        try:
                            if len(price[0].contents) < 11:
                                fee = price[0].contents[0o3].contents[3].contents[0]
                                discount = price[1].contents[0].contents[3].contents[0]
                            else:
                                fee = price[0].contents[0o3].contents[3].contents[0]
                                discount = price[0].contents[9].contents[3].contents[0]
                            fee = fee.replace('\r', '').replace('\n', '')
                            fee = unidecode(fee)
                            fee = fee.replace(",", "").replace('ryl', '').replace(' ', '')
                            discount = discount.replace('\r', '').replace('\n', '')
                            discount = unidecode(discount)
                            discount = discount.replace(",", "").replace('ryl', '').replace(' ', '')
                            row.append(fee)
                            row.append(discount)
                        except:
                            row.append([])
                            row.append([])

                        collar_pattern = r'یقه :(.*?)\آ'
                        if category_name == "shirt" or "dress-shirt":
                            collar = re.findall(collar_pattern, title)
                            if not collar:
                                row.append([])
                            else:
                                row.append(collar[0])
                        elif category_name == 't-shirt':
                            collar = re.findall(collar_pattern, title)
                            if not collar:
                                row.append([])
                            else:
                                row.append(collar[0])
                        else:
                            row.append([])

                        df_lst.append(row)
                        df = pd.DataFrame(df_lst, columns=header)
                        df.to_csv(f'/home/vargha/Desktop/salian_{category_name}_links.csv', columns=header, index=False)
            print(f'page {p} is done')
        print(f'category : {category_name} is done: {len(df_lst)}')

        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/salian_{category_name}_links.csv', columns=header, index=False)
        print(len(df))


if __name__ == '__main__':
    get_salian()
