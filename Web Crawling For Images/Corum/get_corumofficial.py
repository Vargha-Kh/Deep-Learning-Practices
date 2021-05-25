import re

from bs4 import BeautifulSoup
import requests
import pandas as pd
from unidecode import unidecode

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Price', 'Discounted Price', 'Collar', 'Sleeve']


def get_corumofficial():
    categories = {
        "pants": "https://corumofficial.com/product-category/men/category-men-collection/category-men-pants/",
        "shirt": "https://corumofficial.com/product-category/men/category-men-collection/caregory-men-shirt/",
        "t-shirt": "https://corumofficial.com/product-category/men/category-men-collection/t-shirt/"
    }
    base_link = "https://corumofficial.com"
    for category_name, category in categories.items():
        df_lst = []
        page = requests.get(category)
        soup = BeautifulSoup(page.content, 'html.parser')
        divs = soup.find_all('div', {'class': "image-fade_in_back"})
        for div in divs:
            a = div.contents[1]
            href = a['href']
            title = div.contents[1].contents[1]['alt']
            clothes_filters = ['تیشرت', 'شلوار', 'پیراهن', 'تی شرت']
            if any(clothes in title for clothes in clothes_filters):
                row = [title, href]
                page_ = requests.get(href)
                soup_ = BeautifulSoup(page_.content, 'html.parser')
                imgs = soup_.find_all('img', class_='skip-lazy')
                img_links = [img['data-large_image'] for img in imgs]
                row.extend(img_links)
                number_of_pictures = 8 - (len(row) - 2)
                if number_of_pictures != 0:
                    for i in range(number_of_pictures):
                        row.append([])
                price = soup_.find_all('p', class_='price product-page-price')
                fee = price[0].contents[1].contents[0].contents[0].contents[0]
                fee = unidecode(fee)
                fee = fee.replace(",", "")
                fee = int(fee) * 10
                discount = price[0].contents[3].contents[0].contents[0].contents[0]
                discount = unidecode(discount)
                discount = discount.replace(",", "")
                discount = int(discount) * 10
                row.append(fee)
                row.append(discount)
                if category_name == "shirt":
                    collar_pattern = 'یقه(.*$)'
                    collar = re.findall(collar_pattern, title)
                    if not collar:
                        row.append([])
                    else:
                        row.append(collar[0])

                    sleeve_pattern = 'آستین(.*$)'
                    sleeve = re.findall(sleeve_pattern, title)
                    if not sleeve:
                        row.append("بلند")
                    else:
                        row.append(sleeve[0])
                elif category_name == 't-shirt':
                    collar_pattern = 'یقه(.*$)'
                    collar = re.findall(collar_pattern, title)
                    if not collar:
                        row.append([])
                    else:
                        row.append(collar[0])
                    row.append('کوتاه')
                else:
                    row.append([])
                    row.append([])

                print(f"{title} has {len(img_links)} images, link: {href}")
                df_lst.append(row)

        print(f'category : {category_name} is done: {len(df_lst)}')
        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/corumofficial_{category_name}_links.csv', columns=header, header=header,
                  index=False)
        print(len(df))


if __name__ == '__main__':
    get_corumofficial()
