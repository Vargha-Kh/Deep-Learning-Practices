from bs4 import BeautifulSoup
import requests
import pandas as pd
import re

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image',
          'Sixth Image', 'Seventh Image', 'Eighth Image', 'Ninth Image', 'Tenth Image', 'Price', 'Discounted Price',
          'Sleeve', 'Collar']


def get_shixon():
    categories = {
        "shirt": "https://www.shixon.com/categories/men-shirts?sac=60&s=888&p=",
        "pants": "https://www.shixon.com/categories/men-pants?sac=60&s=888&p=",
        "ladies_pants": "https://www.shixon.com/categories/women-trousers-leggings?sac=60&s=888&p=",
        "t-shirt": "https://www.shixon.com/categories/men-tshirts-poloshirts?sac=60&s=888&p=",
    }
    base_link = "https://www.shixon.com"
    for category_name, category in categories.items():
        df_lst = []
        for p in range(1, 100):
            file_link = category + str(p)
            page = requests.get(file_link)
            soup = BeautifulSoup(page.content, 'html.parser')
            div_main_good = soup.find(id='DivMainGood')
            a_links = div_main_good.find_all("a")
            if not a_links:
                break
            for link in a_links:
                if len(link.contents) == 3:
                    clothes_filters = ['تی شرت', 'شلوار', 'پیراهن', 'تیشرت', 'PANT', 'T-SHIRT', 'SHIRT']
                    title = link.contents[1]['alt']
                    if any(clothes in title for clothes in clothes_filters):
                        href = base_link + link['href']
                        row = [title, href]
                        page_ = requests.get(href)
                        soup_ = BeautifulSoup(page_.content, 'html.parser')
                        a_links = soup_.find_all('a', {'class': "DetailColor"})
                        img_links = []
                        for a in a_links:
                            img_link = a.contents[1]['data-thumb']
                            if img_link.endswith('.jpg'):
                                img_links.append(img_link)
                        row.extend(img_links)
                        number_of_pictures = 10 - (len(row) - 2)
                        if number_of_pictures != 0:
                            for i in range(number_of_pictures):
                                row.append([])

                        price = soup_.find_all('span', {'class': 'productPrice'})
                        try:
                            if len(price[0].contents) == 0:
                                fee = price[1].contents[0]
                                discount = price[1].contents[0]
                            else:
                                fee = price[1].contents[0]
                                discount = price[0].contents[0]

                            if fee:
                                fee = fee.replace('\r', '').replace('\n', '')
                                fee = fee.replace(",", "").replace(' ', '')
                                fee = int(fee) * 10
                            if discount:
                                discount = discount.replace('\r', '').replace('\n', '')
                                discount = discount.replace(",", "").replace(' ', '')
                                discount = int(discount) * 10
                            row.append(fee)
                            row.append(discount)
                        except:
                            row.append([])
                            row.append([])

                        details = soup_.find_all('div', {'class': 'panel-body'})
                        details = details[0].text
                        collar_pattern = r'یقه:(\s*>?\S.*\S)'
                        sleeve_pattern = r'آستین(\s*>?\S*)'
                        if category_name == "shirt":
                            sleeve = re.findall(sleeve_pattern, title)
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
                            sleeve = re.findall(sleeve_pattern, title)
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
                        df.to_csv(f'/home/vargha/Desktop/shixon_{category_name}_links.csv', columns=header, index=False)
            print(f'page {p} is done')

        print(f'category : {category_name} is done: {len(df_lst)}')
        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/shixon_{category_name}_links.csv', columns=header, index=False)


if __name__ == '__main__':
    get_shixon()
