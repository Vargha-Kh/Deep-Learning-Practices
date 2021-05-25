from bs4 import BeautifulSoup
import requests
import pandas as pd
from unidecode import unidecode
import re
import time
from selenium import webdriver

driver = webdriver.PhantomJS("/home/vargha/phantomjs-2.1.1-linux-x86_64/bin/phantomjs")

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Ninth Image', 'Tenth Image', 'Price', 'Value of Discount',
          'Collar', 'Sleeve']


def get_digistyle():
    categories = {
        "t-shirt": "https://www.digistyle.com/category-men-tee-shirts-and-polos/",
        "shirt": "https://www.digistyle.com/category-men-shirts",
        "pants": "https://www.digistyle.com/category-men-trousers-jumpsuits",
        "jeans": "https://www.digistyle.com/category-men-jeans"
    }
    base_link = "https://www.digistyle.com"
    # df = []
    for category_name, category in categories.items():
        df_lst = []
        href_list = []
        browser = webdriver.Firefox()
        browser.get(category)
        len_of_page = browser.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);var len_of_page=document.body.scrollHeight;return len_of_page;")
        match = False
        while not match:
            last_count = len_of_page
            time.sleep(10)
            len_of_page = browser.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);var len_of_page=document.body.scrollHeight;return len_of_page;")
            if last_count == len_of_page:
                match = True

        source_data = browser.page_source
        soup = BeautifulSoup(source_data, features='lxml')

        # for p in range(1, 250):
        #     file_link = category + str(p)
        #     try:
        # page = requests.get(category)
        #     except requests.exceptions.ConnectionError:
        #         print(f'connection error {file_link}')
        #         continue
        # soup = BeautifulSoup(page.content, 'html.parser')
        try:
            ul_cloth_list = soup.find_all("div", class_="c-product-grid js-product-grid")[0]
            a_links = ul_cloth_list.find_all('a')
            if not a_links:
                break
            for link in a_links:
                # if len(link.contents) == 3:
                href = link['href']
                title = href[9:]
                img_link = base_link + href
                if href not in href_list:
                    href_list.append(href)
                    clothes_filters = ['تی شرت', 'شلوار', 'پیراهن', 'تیشرت']
                    if any(clothes in title for clothes in clothes_filters):
                        row = [title, img_link]
                        try:
                            page_ = requests.get(img_link)
                        except requests.exceptions.ConnectionError:
                            print(f'connection error {img_link}')
                            continue
                        soup_ = BeautifulSoup(page_.content, 'html.parser')
                        image_divs = soup_.find_all('div', class_="swiper-wrapper")
                        if len(image_divs) == 0:
                            continue
                        else:
                            image_divs = image_divs[0]
                        images = image_divs.find_all('img', class_='c-product-item__image')
                        main = [img for img in images if img['alt'][-1] == '1']
                        if len(main) != 1:
                            continue
                        images.insert(0, main[0])
                        for img in images:
                            img_link = img['src']
                            cursor = img_link.find('.jpg')
                            img_link = img_link[:cursor + 4]
                            if img_link not in row:
                                row.append(img_link)
                        print(len(row) - 1, " images has been added!")
                        number_of_pictures = 10 - (len(row) - 2)
                        if number_of_pictures != 0:
                            for i in range(number_of_pictures):
                                row.append([])
                        price = soup_.find_all('div', {
                            'class': 'c-price-container c-price-container--quick-view-price-original js-rrp-price'})
                        fee = price[0]['data-price-value']
                        fee = unidecode(fee)
                        fee = fee.replace(",", "")
                        fee = int(fee) * 10
                        discount = price[0].next['data-price-value']
                        discount = unidecode(discount)
                        discount = discount.replace(",", "")
                        if any(char.isdigit() for char in discount):
                            discount = ''.join(i for i in discount if i.isdigit())
                            discount = int(discount) * 10
                        else:
                            discount = []
                        row.append(fee)
                        row.append(discount)
                        details = soup_.find_all('ul', {'class': 'c-product__specs-table'})
                        details_text = details[0].text
                        collar_pattern = r'یقه(\s*>?\S.*\s)'
                        sleeve_pattern = r'آستین(\s*>?\S.*\s)'
                        if category_name == "shirt":
                            collar = re.findall(collar_pattern, details_text)
                            if not collar:
                                row.append([])
                            else:
                                collar_text = collar[0].replace('\r', '').replace('\n', '').replace(' ', '')
                                row.append(collar_text)

                            sleeve = re.findall(sleeve_pattern, details_text)
                            if not sleeve:
                                row.append("بلند")
                            else:
                                sleeve_text = sleeve[0].replace('\r', '').replace('\n', '').replace(' ', '')
                                row.append(sleeve_text)
                        elif category_name == 't-shirt':
                            collar = re.findall(collar_pattern, details_text)
                            if not collar:
                                row.append([])
                            else:
                                collar_text = collar[0].replace('\r', '').replace('\n', '').replace(' ', '')
                                row.append(collar_text)
                            sleeve = re.findall(sleeve_pattern, details_text)
                            if not sleeve:
                                row.append("کوتاه")
                            else:
                                sleeve_text = sleeve[0].replace('\r', '').replace('\n', '').replace(' ', '')
                                row.append(sleeve_text)
                        else:
                            row.append([])
                            row.append([])
                        df_lst.append(row)
                        df = pd.DataFrame(df_lst, columns=header)
                        df.to_csv(f'/home/vargha/Desktop/digi_style_{category_name}_links.csv', columns=header,
                                  index=False)
        except:
            pass

        print(f'category : {category_name} is done: {len(df_lst)}')
        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/digi_style_{category_name}_links.csv', columns=header, index=False)


if __name__ == '__main__':
    get_digistyle()
