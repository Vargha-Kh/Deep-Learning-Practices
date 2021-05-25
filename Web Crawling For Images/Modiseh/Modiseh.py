import time
from bs4 import BeautifulSoup
import requests
import pandas as pd
from unidecode import unidecode
import re

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Price', 'Discounted Price',
          'Sleeve', 'Collar']

headers = {'User-Agent': 'firefox'}


def get_modiseh():
    categories = {
        "shirt": "https://www.modiseh.com/men/clothes/shirts?p=",
        "pants": "https://www.modiseh.com/men/clothes/trousers?p=",
        "t-shirt": "https://www.modiseh.com/men/clothes/t-shirts?p="}
    base_link = "https://www.modiseh.com"
    for category_name, category in categories.items():
        df_lst = []
        href_list = []
        for p in range(1, 100):
            file_link = category + str(p)
            page = requests.get(file_link, headers=headers)
            soup = BeautifulSoup(page.content, 'html.parser')
            divs = soup.find_all('div', {'class': "product-item-top"})
            if not divs:
                break
            for div in divs:
                href = div.contents[0]["href"]
                if href not in href_list:
                    href_list.append(href)
                    page_ = requests.get(href, headers=headers)
                    soup_ = BeautifulSoup(page_.content, 'html.parser')
                    title = soup_.contents[2].contents[0].contents[0o7]['content']
                    clothes_filters = ['تی شرت', 'شلوار', 'پیراهن', 'تیشرت', 'PANT', 'T-SHIRT', 'SHIRT']
                    if any(clothes in title for clothes in clothes_filters):
                        row = [title, href]
                        a_links = soup_.find_all("div", {'class': "gallery-placeholder"})
                        if len(a_links) == 0:
                            continue
                        for a in a_links:
                            img_link = a.contents[0]['src']
                            row.append(img_link)
                        print(len(row) - 1, " images has been added!")
                        number_of_pictures = 8 - (len(row) - 2)
                        if number_of_pictures != 0:
                            for i in range(number_of_pictures):
                                row.append([])

                        price = soup_.find_all('span', {'class': 'price-container price-final_price tax weee'})
                        try:
                            if len(price[0].contents) <= 7:
                                fee = price[0].contents[3]["data-price-amount"]
                                discount = price[0].contents[3]["data-price-amount"]
                            else:
                                fee = price[0].contents[0].contents[0].contents[0]
                                discount = price[0].contents[2].contents[0].contents[0]
                            fee = fee.replace('\r', '').replace('\n', '')
                            fee = fee.replace(",", "").replace(' ', '')
                            discount = discount.replace('\r', '').replace('\n', '')
                            discount = discount.replace(",", "").replace(' ', '')
                            fee = int(fee) * 10
                            discount = int(discount) * 10
                            row.append(fee)
                            row.append(discount)
                        except:
                            row.append([])
                            row.append([])

                        details_text = ''
                        details = soup_.find_all('div', {'class': 'value'})
                        details = details[1].contents[0].contents[1].contents[0].contents
                        for detail in details:
                            details_text += detail.text + '\n'
                        collar_pattern = r'یقه:(\s*>?\S*\s)'
                        sleeve_pattern = r'آستین:(\s*>?\S*\s)'
                        if category_name == "shirt":
                            sleeve = re.findall(sleeve_pattern, details_text)
                            if not sleeve:
                                row.append([])
                            else:
                                sleeve_text = sleeve[0].replace('\r', '').replace('\n', '').replace(" ", '')
                                row.append(sleeve_text)
                            collar = re.findall(collar_pattern, details_text)
                            if not collar:
                                row.append([])
                            else:
                                collar_text = collar[0].replace('\r', '').replace('\n', '').replace(" ", '')
                                row.append(collar_text)
                        elif category_name == 't-shirt':
                            sleeve = re.findall(sleeve_pattern, details_text)
                            if not sleeve:
                                row.append('کوناه')
                            else:
                                sleeve_text = sleeve[0].replace('\r', '').replace('\n', '').replace(" ", '')
                                row.append(sleeve_text)
                            collar = re.findall(collar_pattern, details_text)
                            if not collar:
                                row.append([])
                            else:
                                collar_text = collar[0].replace('\r', '').replace('\n', '').replace(" ", '')
                                row.append(collar_text)
                        else:
                            row.append([])
                            row.append([])

                        df_lst.append(row)
            print(f'page {p} is done')
            print(f'category : {category} is done: {len(df_lst)}')

            df = pd.DataFrame(df_lst, columns=header)
            df.to_csv(f'/home/vargha/Desktop/modiseh_{category_name}_links.csv', columns=header, index=False)
        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/modiseh_{category_name}_links.csv', columns=header, header=header, index=False)


if __name__ == '__main__':
    get_modiseh()
