from bs4 import BeautifulSoup
import requests
import pandas as pd
from unidecode import unidecode
import re

base_link = "https://www.rosha.com"

categories = {
    "t-shirts": "https://rosha.ir/men-tees-polos-page",
    "shirts": "https://rosha.ir/men-shirt-page",
    "pants": "https://rosha.ir/men-trousers-shorts-page"
}
header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Price', 'Value of Discount',
          'Collar', 'Sleeve']

for category_name, category in categories.items():
    df_lst = []
    href_list = []
    for p in range(1, 3):
        file_link = category + str(p) + ".html"
        page = requests.get(file_link)
        soup = BeautifulSoup(page.content, 'html.parser')
        a_links = soup.find_all("a", class_="product-image")
        if not a_links:
            break
        for link in a_links:
            href = link['href']
            title = link['title']
            if href not in href_list:
                href_list.append(href)
                clothes_filters = ['تی شرت', 'شلوار', 'پیراهن', 'تیشرت', 'پبراهن', 'blue', 'black', 'grey', 'red', 'purple']
                if any(clothes in title for clothes in clothes_filters):
                    row = [title, href]
                    page_ = requests.get(href)
                    soup_ = BeautifulSoup(page_.text, 'html.parser')
                    images = soup_.find_all('div', {'class': "product-img-box col-sm-5"})
                    if len(images) == 0:
                        continue
                    for img in images:
                        img_links = img.contents[0o1].contents[1].contents
                        img_links = [i for i in img_links if i != '\n']
                        for img_link in img_links:
                            href = img_link.get("href")
                            src = img_link.get("src")
                            if href:
                                row.append(href)
                            else:
                                row.append(src)
                    print(len(row) - 1, " images has been added!")
                    number_of_pictures = 8 - (len(row) - 2)
                    if number_of_pictures != 0:
                        for i in range(number_of_pictures):
                            row.append([])

                    price = soup_.find_all('div', {'class': 'price-box'})
                    try:
                        if len(price[0].contents) <= 3:
                            fee = price[0].contents[1].contents[1].contents[0]
                            discount = price[1].contents[1].contents[1].contents[0]
                        else:
                            fee = price[0].contents[1].contents[3].contents[0]
                            discount = price[0].contents[3].contents[3].contents[0]
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

                    collar_pattern = r'یقه(\s*>?\S*\s)'
                    sleeve_pattern = r'آستین(.*?).\s'
                    if category_name == "shirts":
                        sleeve = re.findall(sleeve_pattern, title)
                        if not sleeve:
                            row.append("بلند")
                        else:
                            row.append(sleeve[0])
                        collar = re.findall(collar_pattern, title)
                        if not collar:
                            row.append([])
                        else:
                            row.append(collar[0])
                    elif category_name == 't-shirts':
                        sleeve = re.findall(sleeve_pattern, title)
                        if not sleeve:
                            row.append([])
                        else:
                            row.append(sleeve[0])
                        collar = re.findall(collar_pattern, title)
                        if not collar:
                            row.append([])
                        else:
                            row.append(collar[0])
                    else:
                        row.append([])
                        row.append([])
                    df_lst.append(row)
                    df = pd.DataFrame(df_lst, columns=header)
                    df.to_csv(f'/home/vargha/Desktop/rosha_{category_name}_links.csv', columns=header,
                              index=False)
        print(f'page {p} is done')
    df = pd.DataFrame(df_lst, columns=header)
    df.to_csv(f'/home/vargha/Desktop/rosha_{category_name}_links.csv', index=False, columns=header)
    print(len(df))
