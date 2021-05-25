import time
from bs4 import BeautifulSoup
import requests
import pandas as pd
from unidecode import unidecode
import re

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Price', 'Discounted Price',
          'Sleeve', 'Collar']


def get_salset():
    categories = {
        # "shirt": "https://salset.com/%D9%85%D8%AD%D8%B5%D9%88%D9%84%D8%A7%D8%AA/%d9%85%d8%af-%d9%88-%d9%84%d8%a8%d8%a7%d8%b3/%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/%d9%84%d8%a8%d8%a7%d8%b3-%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/%d9%be%db%8c%d8%b1%d8%a7%d9%87%d9%86-%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/page/",
        "pants": "https://salset.com/%D9%85%D8%AD%D8%B5%D9%88%D9%84%D8%A7%D8%AA/%d9%85%d8%af-%d9%88-%d9%84%d8%a8%d8%a7%d8%b3/%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/%d9%84%d8%a8%d8%a7%d8%b3-%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/%d8%b4%d9%84%d9%88%d8%a7%d8%b1-%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/page/",
        "t-shirt": "https://salset.com/%D9%85%D8%AD%D8%B5%D9%88%D9%84%D8%A7%D8%AA/%d9%85%d8%af-%d9%88-%d9%84%d8%a8%d8%a7%d8%b3/%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/%d9%84%d8%a8%d8%a7%d8%b3-%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/%d8%aa%db%8c%d8%b4%d8%b1%d8%aa-%d9%be%d9%88%d9%84%d9%88%d8%b4%d8%b1%d8%aa-%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/page/"}
    base_link = "https://www.salset.com"
    for category_name, category in categories.items():
        df_lst = []
        href_list = []
        for p in range(1, 100):
            file_link = category + str(p)
            page = requests.get(file_link)
            soup = BeautifulSoup(page.content, 'html.parser')
            divs = soup.find_all("div", class_="photoOverlay")
            if not divs:
                break
            for div in divs:
                href = div.contents[1]['href']
                title = div.contents[1].contents[3]['alt']
                if href not in href_list:
                    href_list.append(href)
                    clothes_filters = ['تی شرت', 'شلوار', 'پیراهن', 'تیشرت', 'PANT', 'T-SHIRT', 'SHIRT']
                    if any(clothes in title for clothes in clothes_filters):
                        row = [title, href]
                        page_ = requests.get(href)
                        soup_ = BeautifulSoup(page_.content, 'html.parser')
                        a_links = soup_.find_all('a', class_="woocommerce-main-image zoom")
                        if len(a_links) == 0:
                            continue
                        for a in a_links:
                            img_link = a['href']
                            row.append(img_link)
                        print(len(row) - 1, " images has been added!")
                        number_of_pictures = 8 - (len(row) - 2)
                        if number_of_pictures != 0:
                            for i in range(number_of_pictures):
                                row.append([])

                        price = soup_.find_all('p', {'class': 'price'})
                        try:
                            if len(price[0].contents) <= 3:
                                fee = price[0].contents[0].contents[0].contents[0]
                                discount = price[0].contents[2].contents[0].contents[0]
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

                        details = soup_.find_all('div', {
                            'class': 'woocommerce-Tabs-panel woocommerce-Tabs-panel--additional_information panel entry-content wc-tab'})
                        details = details[0].text
                        collar_pattern = r'یقه(\s*>?\S*\s)'
                        sleeve_pattern = r'آستین(\s*>?\S*\s)'
                        if category_name == "shirt":
                            sleeve = re.findall(sleeve_pattern, details)
                            if not sleeve:
                                row.append([])
                            else:
                                sleeve_text = sleeve[0].replace('\r', '').replace('\n', '')
                                row.append(sleeve_text)
                            collar = re.findall(collar_pattern, details)
                            if not collar:
                                row.append([])
                            else:
                                collar_text = collar[0].replace('\r', '').replace('\n', '')
                                row.append(collar_text)
                        elif category_name == 't-shirt':
                            sleeve = re.findall(sleeve_pattern, details)
                            if not sleeve:
                                row.append('کوناه')
                            else:
                                sleeve_text = sleeve[0].replace('\r', '').replace('\n', '')
                                row.append(sleeve_text)
                            collar = re.findall(collar_pattern, details)
                            if not collar:
                                row.append([])
                            else:
                                collar_text = collar[0].replace('\r', '').replace('\n', '')
                                row.append(collar_text)
                        else:
                            row.append([])
                            row.append([])

                        df_lst.append(row)
            print(f'page {p} is done')
            print(f'category : {category} is done: {len(df_lst)}')

            df = pd.DataFrame(df_lst, columns=header)
            df.to_csv(f'/home/vargha/Desktop/salset_{category_name}_links.csv', columns=header, index=False)
        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/salset_{category_name}_links.csv', columns=header, header=header, index=False)


if __name__ == '__main__':
    get_salset()
