# 기본 패키지
import time
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# 크롤링 패키지
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select

total_restaurant = set()
total_response = 0

# 크롤링 함수
def yogiyo_crawling(location):
    global total_restaurant, total_response
    total = pd.DataFrame()
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(executable_path='/opt/ml/final_project/chromedriver', chrome_options=chrome_options) # 크롬드라이버 경로 설정
    
    def go_back_page():
        driver.execute_script("window.history.go(-1)")
        time.sleep(5)
        print('페이지 돌아가기 완료!'+'\n')

    try:
        # url입력
        url = "https://www.yogiyo.co.kr/" # 사이트 입력
        driver.get(url) # 사이트 오픈
        driver.maximize_window() # 전체장
        time.sleep(2) # 2초 지연
        
        # 검색창 선택
        xpath = '''//*[@id="search"]/div/form/input'''  # 검색창
        element = driver.find_element_by_xpath(xpath)
        element.clear()
        time.sleep(2)

        # 검색창 입력
        element.send_keys(location)
        time.sleep(2)

        # 조회버튼 클릭
        search_xpath = '''//*[@id="button_search_address"]/button[2]'''
        driver.find_element_by_xpath(search_xpath).click()
        time.sleep(2)

        # 검색 콤보상자 선택
        try:
            search_selector = '#search > div > form > ul > li:nth-child(3) > a'
            search = driver.find_element_by_css_selector(search_selector)
            search.click()
            time.sleep(2)
        except Exception as e:
            print('정확한 주소를 입력하셨습니다!')
        
        #카테고리 페이지
        cafe_xpath = '''//*[@id="category"]/ul/li[13]/span'''
        category_element = driver.find_element_by_xpath(cafe_xpath)
        driver.execute_script("arguments[0].click();", category_element)
        time.sleep(2)

        # 리뷰 많은순 선택
        select=Select(driver.find_element_by_css_selector("#content > div > div.row.restaurant-list-info > div.list-option > div > select"))
        select.select_by_index(2)
        time.sleep(2)
        print('페이지 불러오기 완료!')
        
        print('Start [ {} ] Crawling...'.format(location))
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 스크롤을 가장 아래로 내린다
        time.sleep(2)
        pre_height = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤 위치 저장

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 스크롤을 가장 아래로 내린다
            time.sleep(2)
            cur_height = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤을 저장한다.
            # 스크롤 다운 후 스크롤 높이 다시 가져옴
            if pre_height == cur_height:
                break
            pre_height = cur_height
        time.sleep(2)
        print('모든 카페 리스트 불러오기 완료!')

        # 카페 받아오기
        restaurant_list=[]
        html = driver.page_source
        html_source = BeautifulSoup(html, 'html.parser')
        restaurant_name = html_source.find_all("div", class_ = "restaurant-name ng-binding") #업체명
        for res in restaurant_name:
            restaurant_list.append(res.string)
        print("카페 리스트 받아오기 완료!")
        print("마지막 카페:", restaurant_list[-1])

        for idx, restaurant_name in enumerate(restaurant_list):

            if restaurant_name in total_restaurant:
                print("이미 있는 카페입니다!")
                continue
            else:
                total_restaurant.add(restaurant_name)

            try:
                '//*[@id="content"]/div/div[5]/div/div/div[1]/div/table/tbody/tr/td[2]/div/div[2]/span[2]/text()'
                customer_xpath = '//*[@id="content"]/div/div[5]/div/div/div['+str(idx+1)+']/div/table/tbody/tr/td[2]/div/div[2]/span[2]'
                custom_src = driver.find_element(by=By.XPATH, value=customer_xpath)
                owner_xpath = '//*[@id="content"]/div/div[5]/div/div/div['+str(idx+1)+']/div/table/tbody/tr/td[2]/div/div[2]/span[3]'
                owner_src = driver.find_element(by=By.XPATH, value=owner_xpath)
                custom_len, owner_len = int(custom_src.text[3:]), int(owner_src.text[6:])

            except Exception as e:
                print("리뷰가 없거나 답글이 없습니다.. 다음 카페로..")
                print(e)
                continue


            print('********** '+restaurant_name+' ( '+str(restaurant_list.index(restaurant_name)+1)
                  +'/'+str(len(restaurant_list))+' 번째) **********')
            print("리뷰수:", custom_len, " 답글수:", owner_len)

            if custom_len//10 > owner_len:
                print("답글이 10%도 안되서 다음 가게로....")
                continue

            total_response += owner_len

            # try:

            try:
                search_selector = '#content > div > div:nth-child(5) > div > div > div:nth-child('+str(idx+1)+') > div'
                search = driver.find_element_by_css_selector(search_selector)
                search.click()
                time.sleep(2)
                print("카페 페이지 접속 완료!")
            except Exception as e:
                print(e)
                print('해당 카페가 없음.. 다음 가게로..')
                continue

            try:
                # 리뷰버튼 클릭
                review_xpath = '''//*[@id="content"]/div[2]/div[1]/ul/li[2]/a'''
                driver.find_element(by=By.XPATH, value=review_xpath).click()
                time.sleep(2)

                # 더보기버튼 클릭
                start_time = time.time()
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

                end_time = time.time()
                print('리뷰 페이지 완료!  ', end_time-start_time)
                time.sleep(2)


                html = driver.page_source
                html_source = BeautifulSoup(html, 'html.parser')

                tastes = []
                quantitys = []
                deliverys = []
                menus = []
                customer_reviews = []
                manager_responses = []

                full_review = html_source.find_all("li", attrs={"class":"list-group-item star-point ng-scope", "ng-repeat":"review in restaurant.reviews"})  # 전체 리뷰 스크립트

                rname = html_source.find("span", class_="restaurant-name ng-binding")  # 업체명
                print(rname.text, '-----카페 이름 확인')
                print("데이터 추출 시작!")
                for reviews in tqdm(full_review):
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

                reviews = pd.DataFrame({'업체명':rname.text, '맛':tastes,'양':quantitys,
                                '배달':deliverys,'주문메뉴':menus, '고객리뷰':customer_reviews, '사장답글':manager_responses})
                
                print("사장님 답글:",len(manager_responses))
                total = pd.concat([total, reviews])
                go_back_page()
                time.sleep(2)

                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 스크롤을 가장 아래로 내린다
                time.sleep(2)
                pre_height = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤 위치 저장

                while True:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 스크롤을 가장 아래로 내린다
                    time.sleep(2)
                    cur_height = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤을 저장한다.
                    # 스크롤 다운 후 스크롤 높이 다시 가져옴
                    if pre_height == cur_height:
                        break
                    pre_height = cur_height
                time.sleep(2)

            except Exception as e:
                print(e)
                print("리뷰 크롤링 실패.. 다음 카페로..")
                go_back_page()
                time.sleep(2)

                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 스크롤을 가장 아래로 내린다
                time.sleep(2)
                pre_height = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤 위치 저장

                while True:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 스크롤을 가장 아래로 내린다
                    time.sleep(2)
                    cur_height = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤을 저장한다.
                    # 스크롤 다운 후 스크롤 높이 다시 가져옴
                    if pre_height == cur_height:
                        break
                    pre_height = cur_height
                time.sleep(2)

    except Exception as e:
        print('@@카페 리스트 페이지 에러@@')
        print(e)
        pass

    # total.to_csv("total_data.csv", index=False, encoding="utf-8-sig")
    print('End of [ {} ] Crawling!'.format(location))
    driver.quit()
    return total

# 메인 크롤링 함수
def start_yogiyo_crawling(location_list):
    global total_response
    df = pd.DataFrame()
    for location in location_list:
        try:
            yogiyo = yogiyo_crawling(location)
            df = pd.concat([df, yogiyo])
            print(location+" 크롤링 완료!")
            print("중간 데이터 총량: ", total_response)
        except Exception as e:
            print('***** '+location+' 에러 발생 *****')
            print(e)
            pass
    df.to_csv("total_data.csv", index=False, encoding="utf-8-sig")

# Chrome webdriver - 요기요 메인 페이지 실행
try:
    temp_df = pd.read_csv('lkm_total_v1.csv')
    total_restaurant = set(temp_df['업체명'].unique())
    print("기존 카페 개수: ",len(total_restaurant))
except Exception as e:
    print("기존 카페가 없습니다..")
    print(e)
loc_list=[
    '서울특별시 서대문구 현저동 101 독립문역',
]

start_yogiyo_crawling(loc_list)
print("데이터 총량: ", total_response)

# Chrome webdriver 종료
# driver.close() 
