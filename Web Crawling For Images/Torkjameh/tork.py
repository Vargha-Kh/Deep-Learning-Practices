from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
from unidecode import unidecode

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Price', 'Discounted Price',
          'Sleeve', 'Collar']


def get_tork():
    categories = {
        "shirt": "https://torkjameh.com/102-%D9%81%D8%B1%D9%88%D8%B4-%D8%A7%DB%8C%D9%86%D8%AA%D8%B1%D9%86%D8%AA%DB%8C-%D9%BE%DB%8C%D8%B1%D8%A7%D9%87%D9%86",
        "ladies_pants": "https://torkjameh.com/18-%D8%AE%D8%B1%DB%8C%D8%AF-%D8%B4%D9%84%D9%88%D8%A7%D8%B1-%D8%B3%D8%A7%D9%BE%D9%88%D8%B1%D8%AA",
        "t-shirt": "https://torkjameh.com/104-%D9%81%D8%B1%D9%88%D8%B4-%D8%A7%DB%8C%D9%86%D8%AA%D8%B1%D9%86%D8%AA%DB%8C-%D8%AA%DB%8C%D8%B4%D8%B1%D8%AA",
        "slash": "https://torkjameh.com/149-%D8%B4%D9%84%D9%88%D8%A7%D8%B1-%D8%A7%D8%B3%D9%84%D8%B4-%D9%88-%D8%B3%D8%AA",
        "pants": "https://torkjameh.com/103-%D9%81%D8%B1%D9%88%D8%B4-%D8%A7%DB%8C%D9%86%D8%AA%D8%B1%D9%86%D8%AA%DB%8C-%D8%B4%D9%84%D9%88%D8%A7%D8%B1-%D9%85%D8%B1%D8%AF%D8%A7%D9%86%D9%87"
    }
    base_link = "https://torkjameh.com"
    for category_name, category_link in categories.items():
        df_lst = []
        page = requests.get(category_link)
        soup = BeautifulSoup(page.content, 'html.parser')
        a_links = soup.find_all('a', class_="product_img_link")
        for a in a_links:
            title = a['title']
            href = a['href']
            clothes_filters = ['تی شرت', 'شلوار', 'پیراهن', 'تیشرت', 'اسلش', 'PANT', 'T-SHIRT', 'SHIRT']
            if any(clothes in title for clothes in clothes_filters):
                row = [title, href]
                try:
                    page_ = requests.get(href)
                except requests.exceptions.ConnectionError:
                    print(f"connection error in {href}")
                    continue
                soup_ = BeautifulSoup(page_.content, 'html.parser')
                ul = soup_.find(id='thumbs_list_frame')
                main_img = soup_.find_all('a', class_="fancybox shown")
                if main_img:
                    main_img = main_img[0]
                else:
                    print(f'no main image found, {category_link}   {href}')
                    continue
                img_links = [main_img['href']]
                if ul is not None:
                    for li in ul.contents:
                        img_href = li.contents[1]['href']
                        if img_href not in img_links:
                            img_links.append(img_href)
                row.extend(img_links)
                print(len(img_links), " image are gathered")
                number_of_pictures = 8 - (len(row) - 2)
                if number_of_pictures != 0:
                    for i in range(number_of_pictures):
                        row.append([])

                price = soup_.find_all('div', {'class': 'price'})
                try:
                    if len(price[0].contents) <= 3:
                        fee = price[0].contents[0].contents[2].contents[0]
                        discount = price[0].contents[0].contents[2].contents[0]
                    else:
                        fee = price[0].contents[0].contents[2].contents[0]
                        discount = price[0].contents[0].contents[2].contents[0]
                    fee = fee.replace('\r', '').replace('\n', '').replace(",", "").replace(' ', '')
                    fee = unidecode(fee)
                    fee = fee.replace('twmn', '')
                    discount = discount.replace('\r', '').replace('\n', '').replace(",", "").replace(' ', '')
                    discount = unidecode(discount)
                    discount = discount.replace('twmn', '')
                    fee = int(fee) * 10
                    discount = int(discount) * 10
                    row.append(fee)
                    row.append(discount)
                except:
                    row.append([])
                    row.append([])

                collar_pattern = r'یقه(\s*>?\S*\s)'
                sleeve_pattern = r'آستین|استین(\s*>?\S*\s)'
                if category_name == "shirt":
                    sleeve = re.findall(sleeve_pattern, title)
                    if not sleeve:
                        row.append([])
                    else:
                        sleeve_text = sleeve[0].replace('\r', '').replace('\n', '')
                        row.append(sleeve_text)
                    collar = re.findall(collar_pattern, title)
                    if not collar:
                        row.append([])
                    else:
                        collar_text = collar[0].replace('\r', '').replace('\n', '')
                        row.append(collar_text)
                elif category_name == 't-shirt':
                    sleeve = re.findall(sleeve_pattern, title)
                    if not sleeve:
                        row.append('کوناه')
                    else:
                        sleeve_text = sleeve[0].replace('\r', '').replace('\n', '')
                        row.append(sleeve_text)
                    collar = re.findall(collar_pattern, title)
                    if not collar:
                        row.append([])
                    else:
                        collar_text = collar[0].replace('\r', '').replace('\n', '')
                        row.append(collar_text)
                else:
                    row.append([])
                    row.append([])
                df_lst.append(row)

                df = pd.DataFrame(df_lst, columns=header)
                df.to_csv(f'/home/vargha/Desktop/torkjameh_{category_name}_links.csv', columns=header, index=False)
        print(f'category : {category_name} is done: {len(df_lst)}')

        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/torkjameh_{category_name}_links.csv', columns=header, index=False)
        print(len(df))


if __name__ == '__main__':
    get_tork()
