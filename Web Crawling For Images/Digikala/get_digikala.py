from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
from unidecode import unidecode

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Ninth Image', 'Tenth Image', 'Eleventh Image',
          'Twelve Image', 'Price',
          'Discounted Percent', 'Collar', 'Sleeve']

headers = {'User-Agent': 'chrome'}


def get_digikala():
    categories = {
        "shirt": "https://www.digikala.com/search/category-men-shirts/?pageno=",
        "men-pants": "https://www.digikala.com/search/category-men-trousers-jumpsuits/?pageno=",
        "t-shirt": "https://www.digikala.com/search/category-men-tee-shirts-and-polos/?pageno=",
        "ladies-tshirt": "https://www.digikala.com/search/category-women-tee-shirts-and-polos/?pageno=",
        "sportswear": "https://www.digikala.com/search/category-men-sportswear/?pageno="
    }
    base_link = "https://www.digikala.com"
    df = []
    for category_name, category in categories.items():
        df_lst = []
        for p in range(1, 1000):
            file_link = category + str(p)
            try:
                page = requests.get(file_link, headers=headers)
            except requests.exceptions.ConnectionError:
                print(f'connection error {file_link}')
                continue
            soup = BeautifulSoup(page.content, 'html.parser')
            a_links = soup.find_all('a', {
                'class': "c-product-box__img c-promotion-box__image js-url js-product-item js-product-url"})
            if not a_links:
                break
            for link in a_links:
                row = []
                href = link['href']
                title = href[20:-6]
                img_link = base_link + href
                clothes_filters = ['تیشرت', 'شلوار', 'پیراهن', 'تی شرت']
                if any(clothes in title for clothes in clothes_filters):
                    if category_name == 't-shirt' and 'آستین بلند' in title:
                        pass
                    else:
                        row = [title, img_link]
                    try:
                        page_ = requests.get(img_link)
                    except requests.exceptions.ConnectionError:
                        print(f'connection error {img_link}')
                        continue
                    soup_ = BeautifulSoup(page_.content, 'html.parser')
                    image_divs = soup_.find_all('div', class_="thumb-wrapper")
                    if len(image_divs) == 0:
                        continue
                    main = [div for div in image_divs if div.contents[0]['alt'][-1] == '1']
                    if len(main) != 1:
                        continue
                    image_divs.insert(0, main[0])
                    for div in image_divs:
                        img_link = div.contents[0]['data-src']
                        cursor = img_link.find('.jpg')
                        img_link = img_link[:cursor + 4]
                        if img_link not in row:
                            row.append(img_link)
                    print(len(row) - 1, " images has been added!")
                    number_of_pictures = 12 - (len(row) - 2)
                    if number_of_pictures != 0:
                        for i in range(number_of_pictures):
                            row.append([])
                    try:
                        price = soup_.find_all('div', class_='c-product__seller-price-info')
                        fee = price[1].contents[0].contents[0]
                        fee = unidecode(fee)
                        fee = fee.replace(",", "").replace('\r', '').replace('\n', '')
                        fee = int(fee) * 10
                        discount = price[1].contents[1].contents[0]
                        discount = unidecode(discount)
                        discount = discount.replace('\r', '').replace('\n', '').replace(' ', '')
                        row.append(fee)
                        row.append(discount)
                    except:
                        row.append([])
                        row.append([])

                    try:
                        details = soup_.find_all('div', {'class': 'c-product__params js-is-expandable'})
                        all_details = details[0].text
                        collar_pattern = r'یقه:(\s*>?\S*\s)'
                        sleeve_pattern = r'آستین:(\s*>?\S*\s)'
                        if category_name == "shirt":
                            collar = re.findall(collar_pattern, all_details)
                            if not collar:
                                row.append([])
                            else:
                                collar_text = collar[0].replace(" ", '').replace('\r', '').replace('\n', '')
                                row.append(collar_text)
                            sleeve = re.findall(sleeve_pattern, all_details)
                            if not sleeve:
                                row.append("بلند")
                            else:
                                sleeve_text = sleeve[0].replace(" ", '').replace('\r', '').replace('\n', '')
                                row.append(sleeve_text)

                        elif category_name == 't-shirt' or 'ladies-tshirt':
                            collar = re.findall(collar_pattern, all_details)
                            if not collar:
                                row.append([])
                            else:
                                collar_text = collar[0].replace(" ", '').replace('\r', '').replace('\n', '')
                                row.append(collar_text)
                            sleeve = re.findall(sleeve_pattern, all_details)
                            if not sleeve:
                                row.append("کوتاه")
                            else:
                                sleeve_text = sleeve[0].replace(" ", '').replace('\r', '').replace('\n', '')
                                row.append(sleeve_text)

                        elif category_name == 'sportswear':
                            collar = re.findall(collar_pattern, all_details)
                            if not collar:
                                row.append([])
                            else:
                                collar_text = collar[0].replace(" ", '').replace('\r', '').replace('\n', '')
                                row.append(collar_text)
                            sleeve = re.findall(sleeve_pattern, all_details)
                            if not sleeve:
                                row.append([])
                            else:
                                sleeve_text = sleeve[0].replace(" ", '').replace('\r', '').replace('\n', '')
                                row.append(sleeve_text)
                        else:
                            row.append([])
                            row.append([])
                    except:
                        row.append([])
                        row.append([])

                    df_lst.append(row)
                    df = pd.DataFrame(df_lst, columns=header)
                    df.to_csv(f'/home/vargha/Desktop/digikala_{category_name}_links.csv', columns=header, header=header,
                              index=False)
        print(f'category : {category_name} is done: {len(df_lst)}')
        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/digikala_{category_name}_links.csv', columns=header, header=header,
                  index=False)


if __name__ == '__main__':
    get_digikala()
