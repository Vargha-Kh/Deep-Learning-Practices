from bs4 import BeautifulSoup
import requests
import pandas as pd
import re

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Ninth Image', 'Tenth Image', 'Eleventh Image',
          'Twelve Image', 'Thirteenth Image', 'Fourteenth Image', 'Fifteenth', 'Sixteenth', 'Seventeenth', 'Eighteenth',
          'Price', 'Discounted Price', 'Sleeve']

categories = {
    "t-shirt": "https://riskonline.ir/product-category/%d8%aa%db%8c%d8%b4%d8%b1%d8%aa/page/",
    "shirt": "https://riskonline.ir/product-category/%d9%be%db%8c%d8%b1%d8%a7%d9%87%d9%86/page/",
    "men_pants": "https://riskonline.ir/product-category/%d8%b4%d9%84%d9%88%d8%a7%d8%b1/page/",
}


def get_risk():
    for category_name, category_link in categories.items():
        df_lst = []
        for p in range(1, 1000):
            file_link = category_link + str(p)
            try:
                page = requests.get(file_link)
            except requests.exceptions.ConnectionError:
                print(f'connection error {file_link}')
                continue
            soup = BeautifulSoup(page.content, 'html.parser')
            # imgs = soup.find_all("img",
            #                      class_="attachment-woocommerce_thumbnail size-woocommerce_thumbnail")
            hs = soup.find_all("h3", class_="product-title")
            divs = soup.find_all('div', class_="product-element-top")
            if len(hs) == 0:
                break
            if len(divs) == 0:
                continue
            for h, a in zip(hs, divs):
                img = a.find("img", class_="attachment-woocommerce_thumbnail size-woocommerce_thumbnail")
                href = a.contents[1]['href']
                title = h.contents[0].contents[0]
                src = img['src'][:-12] + ".jpg"
                clothes_filters = ['تی شرت', 'شلوار', 'پیراهن', 'تیشرت']
                if any(clothes in title for clothes in clothes_filters):
                    row = [title, href, src]
                    try:
                        href.replace("'", '')
                        page_ = requests.get(href).text
                    except requests.exceptions.ConnectionError:
                        print(f'connection error {href}')
                        continue
                    soup_ = BeautifulSoup(page_, 'html5lib')
                    images = soup_.find_all('div', {'class': 'col-12'})
                    lists_of_contents = images[0o3].contents[1].contents
                    if len(lists_of_contents) > 0:
                        img_links = [img.contents[0].contents[0]['href'] for img in lists_of_contents[1:-1]]
                        row.extend(img_links)
                    number_of_pictures = 18 - (len(row) - 2)
                    if number_of_pictures != 0:
                        for i in range(number_of_pictures):
                            row.append([])
                    price = soup_.find_all('p', {'class': 'price'})
                    try:
                        if len(price[0].contents) <= 1:
                            fee = price[0].contents[0].contents[0].contents[0]
                            discount = price[0].contents[0].contents[0].contents[0]
                        else:
                            fee = price[0].contents[0].contents[0].contents[0].contents[0]
                            discount = price[0].contents[2].contents[0].contents[0].contents[0]

                        fee = fee.replace('\r', '').replace('\n', '')
                        fee = fee.replace(",", "")
                        discount = discount.replace('\r', '').replace('\n', '')
                        discount = discount.replace(",", "")
                        fee = int(fee) * 10
                        discount = int(discount) * 10
                    except:
                        fee = []
                        discount = []
                    row.append(fee)
                    row.append(discount)
                    detail = soup_.find('span', {'class': 'posted_in'})
                    if category_name == "shirt":
                        sleeve_pattern = r'آستین(.*?).\s'
                        sleeve = re.findall(sleeve_pattern, detail.text)
                        if not sleeve:
                            row.append("بلند")
                        else:
                            row.append(sleeve[0])
                    elif category_name == 't-shirt':
                        sleeve_pattern = r'آستین(.*?).\s'
                        sleeve = re.findall(sleeve_pattern, detail.text)
                        if not sleeve:
                            row.append("کوتاه")
                        else:
                            row.append(sleeve[0])
                    else:
                        row.append([])
                    df_lst.append(row)
            print(f'page {p} of category {category_name} is done')
        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/risk_{category_name}_links.csv', header=header, columns=header, index=False)
        print(f"{category_name} is done!")


if __name__ == '__main__':
    get_risk()
