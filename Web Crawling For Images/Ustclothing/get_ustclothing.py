from bs4 import BeautifulSoup
import requests
import pandas as pd

header = ['Name', 'Main Web Page', 'Main Image', 'Second Image', 'Third Image', 'Fourth Image', 'Fifth Image', 'Price', "Discounted price"]


def get_ustclothing():
    categories = {
        "t-shirt": "http://www.ustclothing.com/product-category/%d9%85%d8%ad%d8%b5%d9%88%d9%84%d8%a7%d8%aa/page/"}
    base_link = "https://www.ustclothing.com"
    for category_name, category in categories.items():
        df_lst = []
        for p in range(1, 100):
            row = []
            file_link = category + str(p)
            page = requests.get(file_link)
            soup = BeautifulSoup(page.content, 'html.parser')
            a_links = soup.find_all("a", class_="featured-image")
            a_headers = soup.find_all('a', class_='woocommerce-LoopProduct-link woocommerce-loop-product__link')

            if not a_links:
                break
            for a, a_head in zip(a_links, a_headers):
                img = a.find_all('img')[0]
                href = a['href']
                title = a_head.contents[0].contents[0]
                clothes_filters = ['تی شرت', 'شلوار', 'پیراهن', 'تیشرت', 'PANT', 'T-SHIRT', 'SHIRT']
                if any(clothes in title for clothes in clothes_filters):
                    row = [title, href, img['srcset'].split(',')[-1].split(' ')[1]]
                    print(len(row) - 1, " images has been added!")
                    number_of_pictures = 5 - (len(row) - 2)
                    if number_of_pictures != 0:
                        for i in range(number_of_pictures):
                            row.append([])

                    page_ = requests.get(href)
                    soup_ = BeautifulSoup(page_.content, 'html.parser')
                    price = soup_.find_all('p', {'class': 'price'})
                    try:
                        if len(price[0].contents) < 2:
                            fee = price[0].contents[0].contents[0]
                            discount = price[0].contents[0].contents[0]
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
                    df_lst.append(row)
                    print(f'page {p} is done')
                    df = pd.DataFrame(df_lst, columns=header)
                    df.to_csv(f'/home/vargha/Desktop/ustclothing_{category_name}_links.csv', index=False, columns=header)
        print(f'category : {category_name} is done: {len(df_lst)}')
        df = pd.DataFrame(df_lst, columns=header)
        df.to_csv(f'/home/vargha/Desktop/ustclothing_{category_name}_links.csv', index=False, columns=header)


if __name__ == '__main__':
    get_ustclothing()
