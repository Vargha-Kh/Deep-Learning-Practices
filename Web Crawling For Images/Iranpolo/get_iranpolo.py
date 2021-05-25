from bs4 import BeautifulSoup
import pandas as pd
import requests  # to get image from the web
from unidecode import unidecode
import re

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Price', 'Discounted Price', 'Collar', 'Sleeve']


def get_iranpolo():
    categories = {
        "pants": ("https://www.iranuspolo.com/men/trousers.html?ajaxcatalog=true&p=", 6),
        # "shirt": ("https://www.iranuspolo.com/men/shirt.html?ajaxcatalog=true&p=", 21),
        # "t-shirt": ("https://www.iranuspolo.com/men/tshirt.html?ajaxcatalog=true&p=", 21)
    }
    base_link = "https://www.iranpolo.com"
    save_path = './images/'
    for category_name, (category, p_n) in categories.items():
        df_lst = []
        for p in range(1, p_n + 1):
            file_link = category + str(p)
            page = requests.get(file_link)
            soup = BeautifulSoup(page.content, 'html.parser')
            a_links = soup.find_all("a", class_="product-image")
            if not a_links:
                break
            for a in a_links:
                href = a['href']
                title = a['title']
                clothes_filters = ['تیشرت', 'شلوار', 'پیراهن', 'تی شرت']
                if any(clothes in title for clothes in clothes_filters):
                    row = [title, href]
                    try:
                        page_ = requests.get(href)
                    except requests.exceptions.ConnectionError:
                        print(f"connection error in {href}")
                        continue

                    soup_ = BeautifulSoup(page_.content, 'html.parser')
                    divs = soup_.find_all('div', class_='MagicToolboxSelectorsContainer')
                    if len(divs) == 0:
                        continue
                    else:
                        a_inner_links = divs[0].contents[1].find_all('a')
                    main_a = soup_.find_all('a', class_='MagicZoom')[0]
                    row.append(main_a['href'])
                    for index, img in enumerate(a_inner_links):
                        img_link = img['href']
                        if img_link not in row:
                            row.append(img_link)
                    print(len(row) - 1, " images has been added!")
                    number_of_pictures = 8 - (len(row) - 2)
                    if number_of_pictures != 0:
                        for i in range(number_of_pictures):
                            row.append([])
                    price = soup_.find_all('span', class_='price')
                    fee = price[00].contents[0]
                    fee = fee.replace("'", "").replace('\r', '').replace('\n', '')
                    fee = unidecode(fee)
                    fee = fee.replace(",", "").replace('ryl', '')
                    discount = price[0o1].contents[0]
                    discount = discount.replace("'", "").replace('\r', '').replace('\n', '')
                    discount = unidecode(discount)
                    discount = discount.replace(",", "").replace('ryl', '')
                    row.append(fee)
                    row.append(discount)
                    if category_name == "shirt":
                        collar_pattern = 'یقه(.*$)'
                        collar = re.findall(collar_pattern, title)
                        if not collar:
                            row.append([])
                        else:
                            row.append(collar[0])

                        sleeve_pattern = r'آستین(\s*>?\S....)'
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
                    df_lst.append(row)

                print(f'page {p} is done')
            print(f'category : {category_name} is done: {len(df_lst)}')

            df = pd.DataFrame(df_lst, columns=header)
            df.to_csv(f'/home/vargha/Desktop/iranpolo_{category_name}_links.csv', columns=header, index=False)
            print(len(df))


if __name__ == '__main__':
    get_iranpolo()
