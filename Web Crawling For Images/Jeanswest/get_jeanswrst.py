from bs4 import BeautifulSoup
import requests
import pandas as pd
from unidecode import unidecode
import re

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Ninth Image', 'Tenth Image', 'Eleventh Image',
          'Twelve Image', 'Thirteenth Image', 'Fourteenth Image', 'Fifteenth', 'Sixteenth', 'Seventeenth', 'Eighteenth',
          'Price', 'Discounted Price', 'Collar', 'Sleeve', 'Brand']


def get_jeanswest():
    categories = {
        # "men_pants": "https://www.jeanswest.ir/catalog/category/8/?pid=",
        "ladies_pants": "https://www.jeanswest.ir/catalog/category/24/?pid=",
        "shirt": "https://www.jeanswest.ir/catalog/category/11/?pid=",
        "t-shirt": "https://www.jeanswest.ir/catalog/category/832/?pid="}
    base_link = "https://www.jeanswest.ir"
    for category_name, category in categories.items():
        df_lst = []
        for p in range(1, 1000):
            file_link = category + str(p)
            page = requests.get(file_link)
            soup = BeautifulSoup(page.content, 'html.parser')
            divs = soup.find_all("div", class_="product_image")
            if not divs:
                break
            for div in divs:
                href = div.contents[1]['href']
                title = div.contents[1].contents[3]['alt']
                clothes_filters = ['تیشرت', 'شلوار', 'پیراهن', 'تی شرت']
                if any(clothes in title for clothes in clothes_filters):
                    row = [title, href]
                    page_ = requests.get(href)
                    soup_ = BeautifulSoup(page_.content, 'html.parser')
                    img_links = soup_.find_all('img', class_="product_image_box")
                    if len(img_links) == 0:
                        continue
                    for img in img_links:
                        if img['alt'] == title and 'thickbox_default' in img['src']:
                            img_link = img['src']
                            row.append(img_link)
                    print(len(row) - 1, " images has been added!")
                    number_of_pictures = 18 - (len(row) - 2)
                    if number_of_pictures != 0:
                        for i in range(number_of_pictures):
                            row.append([])
                    try:
                        price = soup_.find_all('div', class_='product-price jeans-r-flex product-info')
                        fee = price[0].contents[1].contents[0]
                        if fee == '\n':
                            fee = price[0].contents[3].contents[0].contents[0]
                        fee = fee.replace(",", "").replace('\r', '').replace('\n', '')
                        fee = unidecode(fee)
                        fee = fee.replace(",", "").replace('twmn', '')
                        fee = int(fee) * 10
                        discount = price[0].contents[3].contents[0].contents[0]
                        discount = discount.replace(",", "").replace('\r', '').replace('\n', '')
                        discount = unidecode(discount)
                        discount = discount.replace(",", "").replace('twmn', '')
                        discount = int(discount) * 10
                        row.append(fee)
                        row.append(discount)
                    except:
                        row.append([])
                        row.append([])
                        pass
                    try:
                        details = soup_.find_all('div', {'class': "jeans-column-flex"})
                        all_details = [detail.text for detail in details]
                        all_details = ' '.join(all_details)
                        if category_name == "shirt":
                            collar_pattern = 'یقه :(.*?)\آ'
                            collar = re.findall(collar_pattern, all_details)
                            if not collar:
                                row.append([])
                            else:
                                row.append(collar[0])

                            sleeve_pattern = r'آستین :(.*?.\s)'
                            sleeve = re.findall(sleeve_pattern, all_details)
                            if not sleeve:
                                row.append("بلند")
                            else:
                                if 'بلند' in sleeve[0]:
                                    row.append("بلند")
                                elif 'کوتاه' in sleeve[0]:
                                    row.append("کوتاه")
                                else:
                                    row.append([])
                            brand_pattern = r'برند(\s*>?\s.*\w)'
                            brand = re.findall(brand_pattern, all_details)
                            if not brand:
                                row.append([])
                            else:
                                if 'Jeanswest' in brand[0]:
                                    row.append("Jeanswest")
                                elif 'JootiJeans' in brand[0]:
                                    row.append("JootiJeans")
                                else:
                                    row.append([])
                        elif category_name == 't-shirt':
                            collar_pattern = 'یقه :(.*?)\آ'
                            collar = re.findall(collar_pattern, all_details)
                            if not collar:
                                row.append([])
                            else:
                                row.append(collar[0])

                            sleeve_pattern = r'آستین :(.*?.\s)'
                            sleeve = re.findall(sleeve_pattern, all_details)
                            if not sleeve:
                                row.append("کوتاه")
                            else:
                                if 'بلند' in sleeve[0]:
                                    row.append("بلند")
                                elif 'کوتاه' in sleeve[0]:
                                    row.append("کوتاه")
                                else:
                                    row.append([])

                            brand_pattern = r'برند(\s*>?\s.*\w)'
                            brand = re.findall(brand_pattern, all_details)
                            if not brand:
                                row.append([])
                            else:
                                if 'Jeanswest' in brand[0]:
                                    row.append("Jeanswest")
                                elif 'JootiJeans' in brand[0]:
                                    row.append("JootiJeans")
                                else:
                                    row.append([])
                        else:
                            row.append([])
                            row.append([])
                            brand_pattern = r'برند(\s*>?\s.*\w)'
                            brand = re.findall(brand_pattern, all_details)
                            if not brand:
                                row.append([])
                            else:
                                if 'Jeanswest' in brand[0]:
                                    row.append("Jeanswest")
                                elif 'JootiJeans' in brand[0]:
                                    row.append("JootiJeans")
                                else:
                                    row.append([])

                    except:
                        row.append([])
                        row.append([])
                        row.append([])
                        pass
                    if len(row) > 21:
                        print(row)
                    df_lst.append(row)
                    print(f'page {p} is done')
                    print(f'category : {category_name} is done: {len(df_lst)}')

        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/jeanswest_{category_name}_links.csv', index=False, columns=header,
                  header=header)


if __name__ == '__main__':
    get_jeanswest()
