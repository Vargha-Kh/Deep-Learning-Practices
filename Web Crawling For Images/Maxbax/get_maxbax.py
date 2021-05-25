from bs4 import BeautifulSoup
import requests
import pandas as pd
from unidecode import unidecode
import re

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Price', 'Discounted Percent', 'Collar', 'Sleeve']

headers = {'User-Agent': 'chrome'}

categories = {
    "shirt": "https://maxbax.com/product-category/%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/%d9%be%db%8c%d8%b1%d8%a7%d9%87%d9%86/page/",
    "t-shirt": "https://maxbax.com/product-category/%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/%d8%aa%db%8c%d8%b4%d8%b1%d8%aa/page/",
    "pants": "https://maxbax.com/product-category/%d9%85%d8%b1%d8%af%d8%a7%d9%86%d9%87/%d8%b4%d9%84%d9%88%d8%a7%d8%b1-%d9%88-%d8%a7%d8%b3%d9%84%d8%b4/page"
}

for category_name, category in categories.items():
    df_lst = []
    href_list = []
    for p in range(1, 100):
        file_link = category + str(p) + "/"
        page = requests.get(file_link, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        links = soup.find_all('div', {'class': 'products__item-img-color-wrapper'})
        if not links:
            break
        for link in links:
            a = link.contents[1]
            href = a['href']
            title = link.contents[1].contents[1]['alt']
            clothes_filters = ['تیشرت', 'شلوار', 'پیراهن', 'تی شرت', 'اسلش']
            if href not in href_list:
                href_list.append(href)
                if any(clothes in title for clothes in clothes_filters):
                    row = [title, href]
                    page_ = requests.get(href, headers=headers)
                    soup_ = BeautifulSoup(page_.text, 'lxml')
                    imgs = soup_.find_all('div', class_='woocommerce-product-gallery__image')
                    img_links = [img.get('data-large_image') for img in imgs[0]]
                    row.extend(img_links)
                    number_of_pictures = 6 - (len(row) - 2)
                    if number_of_pictures != 0:
                        for i in range(number_of_pictures):
                            row.append([])

                    try:
                        if len(link.contents[4].contents[5].contents[1]) < 6:
                            fee = link.contents[4].contents[5].contents[1].contents[1].contents[0].contents[0].contents[
                                0]
                            discount = \
                            link.contents[4].contents[5].contents[1].contents[3].contents[0].contents[0].contents[0]
                        else:
                            fee = link.contents[4].contents[5].contents[1].contents[1].contents[0].contents[0].contents[
                                0]
                            discount = \
                            link.contents[4].contents[5].contents[1].contents[3].contents[0].contents[0].contents[0]
                        fee = fee.replace('\r', '').replace('\n', '')
                        fee = unidecode(fee)
                        fee = fee.replace(",", "").replace(' ', '')
                        fee = int(fee) * 10
                        discount = discount.replace('\r', '').replace('\n', '')
                        discount = unidecode(discount)
                        discount = discount.replace(",", "").replace(' ', '')
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
                    sleeve_pattern = r'/آستین(\s*>?\S*)/m'
                    if category_name == "shirt":
                        sleeve = re.findall(sleeve_pattern, details)
                        if not sleeve:
                            row.append([])
                        else:
                            sleeve_text = sleeve[0].replace('\r', '').replace('\n', '').replace(" ", '')
                            row.append(sleeve_text)
                        collar = re.findall(collar_pattern, details)
                        if not collar:
                            row.append([])
                        else:
                            collar_text = collar[0].replace('\r', '').replace('\n', '').replace(" ", '')
                            row.append(collar_text)
                    elif category_name == 't-shirt':
                        sleeve = re.findall(sleeve_pattern, details)
                        if not sleeve:
                            row.append('کوناه')
                        else:
                            sleeve_text = sleeve[0].replace('\r', '').replace('\n', '').replace(" ", '')
                            row.append(sleeve_text)
                        collar = re.findall(collar_pattern, details)
                        if not collar:
                            row.append([])
                        else:
                            collar_text = collar[0].replace('\r', '').replace('\n', '').replace(" ", '')
                            row.append(collar_text)
                    else:
                        row.append([])
                        row.append([])
                    df_lst.append(row)
                    df = pd.DataFrame(df_lst, columns=header)
                    df.to_csv(f'/home/vargha/Desktop/maxbax{category_name}links.csv', index=False, columns=header)
        print(f'page {p} is done')
    df = pd.DataFrame(df_lst, columns=header)
    df.to_csv(f'/home/vargha/Desktop/maxbax{category_name}links.csv', index=False, columns=header)
    print(len(df))
