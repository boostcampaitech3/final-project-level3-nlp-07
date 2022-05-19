from selenium import webdriver
import pandas as pd
from bs4 import BeautifulSoup
import re
from selenium.webdriver.common.by import By
import time
from tqdm import tqdm

# 크롤링 할 사이트의 URL 리스트

url_list = [
    'https://www.yogiyo.co.kr/mobile/#/270572/',
    'https://www.yogiyo.co.kr/mobile/#/446193/',
    'https://www.yogiyo.co.kr/mobile/#/526415/'
]

# chromedriver가 저장된 위치
driver_dir = 'C:\\forCrawl\\chromedriver.exe'

# 결과물을 저장할 위치
data_dir = 'C:\\forCrawl\\'

driver = webdriver.Chrome(driver_dir) # 크롬드라이버 경로 설정

csv_num = 0 # csv 파일명 변수

# 크롤링 시작
for url in tqdm(url_list):
    driver.get(url) # 사이트 오픈
    time.sleep(10)

    # 리뷰버튼 클릭
    review_xpath = '''//*[@id="content"]/div[2]/div[1]/ul/li[2]/a'''
    driver.find_element(by=By.XPATH, value=review_xpath).click()
    time.sleep(3)

    # 더보기
    while True:
        try:
            css_selector = '#review > li.list-group-item.btn-more > a'
            more_reviews = driver.find_element(by=By.CSS_SELECTOR, value=css_selector)
            more_reviews.click()
            time.sleep(2)
        except:
            break
    
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 스크롤을 가장 아래로 내린다
    time.sleep(2)
    pre_height = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤 위치 저장


    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 스크롤을 가장 아래로 내린다
        time.sleep(1)
        cur_height = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤을 저장한다.
        # 스크롤 다운 후 스크롤 높이 다시 가져옴
        if pre_height == cur_height:
            break
        # pre_height == cur_height
        pre_height = cur_height

    time.sleep(3)

    # 페이지 소스 출력
    html = driver.page_source
    html_source = BeautifulSoup(html, 'html.parser')

    # csv 파일 만들기 위한 list 설정
    tastes = []
    quantitys = []
    deliverys = []
    menus = []
    customer_reviews = []
    manager_responses = []

    full_review = html_source.find_all("li", attrs={"class":"list-group-item star-point ng-scope", "ng-repeat":"review in restaurant.reviews"})  # 전체 리뷰 스크립트

    restaurant_name = html_source.find("span", class_="restaurant-name ng-binding")  # 업체명

    for reviews in full_review:
        # 데이터 추출
        manager_response = reviews.find_all("p", attrs={"class": "ng-binding", "ng-bind-html": "review.owner_reply.comment|strip_html"})

        if len(manager_response) == 0: # 사장님 답글이 없을 경우 폐기
            continue

        taste = reviews.find_all("span", attrs={"class": "points ng-binding", "ng-show": "review.rating_taste > 0"})
        quantity = reviews.find_all("span", attrs={"class": "points ng-binding", "ng-show": "review.rating_quantity > 0"})
        delivery = reviews.find_all("span", attrs={"class": "points ng-binding", "ng-show": "review.rating_delivery > 0"})
        menu = reviews.find_all("div", class_="order-items default ng-binding")
        customer_review = reviews.find_all("p", attrs={"class": "ng-binding", "ng-show": "review.comment"})

        
        tastes.append(taste[0].string if taste else '별점X')  # 별점-맛
        quantitys.append(quantity[0].string if quantity else '별점X')  # 별점-양
        deliverys.append(delivery[0].string if delivery else '별점X')  # 별점-배달
        menus.append(menu[0].string if menu else '메뉴X')  # 주문 메뉴
        customer_reviews.append(customer_review[0].string)  # 고객 리뷰
        manager_responses.append(manager_response[0].string)  # 사장님 답글
    
    time.sleep(10) # 크롤링 소요시간 임의 설정

    # csv 파일로 저장하기
    reviews = pd.DataFrame({'업체명':restaurant_name, '맛':tastes,'양':quantitys,
                            '배달':deliverys,'주문메뉴':menus, '고객리뷰':customer_reviews, '사장답글':manager_responses})
    
    reviews.to_csv(data_dir + f"review_data{csv_num}.csv", index=False, encoding="utf-8-sig")
    csv_num += 1

driver.close()  # 크롬드라이버 종료

# 최종 데이터 병합 시작
final_df = pd.read_csv(data_dir + "review_data0.csv")

for i in tqdm(range(1, csv_num)):
    temp = pd.read_csv(data_dir + f'review_data{i}.csv')
    
    final_df = pd.concat([temp, final_df])

final_df.to_csv(data_dir + "final_review_data.csv", index=False, encoding="utf-8-sig")