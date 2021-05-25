import time
from bs4 import BeautifulSoup
import requests
import pandas as pd
from unidecode import unidecode
import re

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Ninth Image', 'Tenth Image', 'Eleventh Image',
          'Twelve Image', 'Thirteenth Image', 'Fourteenth Image', 'Price', 'Value of Discount',
          'Sleeve']


def get_mavi():
    categories = {
        # "men_pants": "https://www.iranmavi.com/men/trousers.html?p=",
        # "ladies_pants": "https://www.iranmavi.com/women/trousers.html?p=",
        # "shirt": "https://www.iranmavi.com/men/shirts.html?p=",
        "t-shirt": "https://www.iranmavi.com/men/t-shirt.html?p="}
    base_link = "https://www.iranmavi.com"
    for category_name, category in categories.items():
        df_lst = []
        for p in range(1, 10):
            file_link = category + str(p)
            page = requests.get(file_link)
            soup = BeautifulSoup(page.content, 'html.parser')
            a_links = soup.find_all("a", class_="product-image")
            if not a_links:
                break
            for a in a_links:
                href = a['href']
                title = a['title']
                clothes_filters = ['تی شرت', 'شلوار', 'پیراهن']
                if any(clothes in title for clothes in clothes_filters):
                    row = [title, href]
                    page_ = requests.get(href)
                    soup_ = BeautifulSoup(page_.content, 'html.parser')
                    divs = soup_.find_all('div', class_='MagicToolboxSelectorsContainer')
                    if len(divs) == 0:
                        continue
                    else:
                        a_inner_links = divs[0].contents[1].find_all('a')
                    zooms = soup_.find_all('a', class_='MagicZoom')
                    if len(zooms) == 0:
                        continue
                    else:
                        zoom_a = zooms[0]
                        row.append(zoom_a['href'])
                    for img in a_inner_links:
                        img_link = img['href']
                        row.append(img_link)
                    print(len(row) - 1, " images has been added!")
                    number_of_pictures = 14 - (len(row) - 2)
                    if number_of_pictures != 0:
                        for i in range(number_of_pictures):
                            row.append([])
                    price = soup_.find_all('div', class_='price-box')
                    if len(price[0].contents) <= 3:
                        fee = price[0].contents[1].contents[1].contents[0]
                        discount = price[1].contents[1].contents[1].contents[0]
                    else:
                        fee = price[0].contents[1].contents[3].contents[0]
                        discount = price[0].contents[3].contents[3].contents[0]
                    fee = fee.replace("'", "").replace('\r', '').replace('\n', '')
                    fee = unidecode(fee)
                    fee = fee.replace(",", "").replace('ryl', '')
                    discount = discount.replace("'", "").replace('\r', '').replace('\n', '')
                    discount = unidecode(discount)
                    discount = discount.replace(",", "").replace('ryl', '')
                    row.append(fee)
                    row.append(discount)
                    if category_name == "shirt":
                        sleeve_pattern = 'آستین(.*$)'
                        sleeve = re.findall(sleeve_pattern, title)
                        if not sleeve:
                            row.append("بلند")
                        else:
                            row.append(sleeve[0])
                    elif category_name == 't-shirt':
                        sleeve_pattern = 'آستین(.*$)'
                        sleeve = re.findall(sleeve_pattern, title)
                        if not sleeve:
                            row.append("کوتاه")
                        else:
                            row.append(sleeve[0])
                    else:
                        row.append([])

                    df_lst.append(row)
            print(f'page {p} is done')
        print(f'category : {category_name} is done: {len(df_lst)}')
        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/iranmavi_{category_name}_links.csv', columns=header, header=header,
                  index=False)
        print(len(df))


if __name__ == '__main__':
    get_mavi()
