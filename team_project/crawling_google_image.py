from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
import time 
import os 
import urllib.request 
from multiprocessing import Pool 
import pandas as pd

keyword = pd.read_csv('./food.csv')
keyword = keyword['음식명']
 
def createFolder(directory): # os를 임포트 하면 폴더 만들 수 있음
    try: 
        if not os.path.exists(directory): #고구마 라는 폴더가 없으면 만들고, 있으면 넘어감
            os.makedirs(directory) 
    except OSError: 
        print ('Error: Creating directory. ' + directory)

def image_download(keyword): 
    createFolder('./'+keyword)
    chromedriver = 'c:/Program Files/Google/Chrome/chromedriver' 
    driver = webdriver.Chrome(chromedriver) 
    driver.implicitly_wait(time_to_wait=3)  # 3초동안 페이지가 로딩되는 걸 기다려준다는 의미
    '''
    implicitly wait : 웹페이지 전체가 넘어올때까지 기다리기
    explicitly wait : 웹페이지의 일부분이 나타날때까지 기다리기
    '''


################################### 구글 이미지 검색 접속 및 검색어 입력 ###################################
    print(keyword, '검색')  # 키워드 검색하도록 명령
    driver.get('https://www.google.co.kr/imghp?hl=ko')  # 크롤링할 URL입력
    Keyword=driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input') # xpath의 원하는 값을 가져옴
    
    '''
    find_element_by_xpath 
    /: 절대 경로를 나타냄
    //: 문서내에서 검색
    //@href : href속성이 있는 모든 태그 선택
    //a[@href='http://google.com']: a 태그의 href 속성에 http://google.com 속상값을 가진 모든 태그 선택
    (//a)[3]: 문서의 세 번째 링크 선택
    (//table)[last()]: 문서의 마지막 테이블 선택
    (//a)[position()< 3]: 문서의 처음 두링크선택
    //table/tr/* : 모든 테이블에서 모든 자식 tr 태그 선택
    //div[@*] : 속성이 하나라도 있는 div 태그 선택
    '''

    Keyword.send_keys(keyword)  # 키워드 입력
    driver.find_element_by_xpath('//*[@id="sbtc"]/button').click()  # 검색 버튼 누르기


################################################# 스크롤 #################################################
    print(keyword+' 스크롤 중 .............') 
    elem = driver.find_element_by_tag_name("body") 
    for i in range(1): 
        elem.send_keys(Keys.PAGE_DOWN)  # 무한 스크롤
        time.sleep(0.1) 
    try: 
        driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[4]/div[2]/input').click()  # 결과 더보기 버튼 누르기
        for i in range(1): 
            elem.send_keys(Keys.PAGE_DOWN) 
            time.sleep(0.1) 
    except: 
        pass 


############################################### 이미지 개수 ###############################################
    images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd") 
    print(keyword+' 찾은 이미지 개수:', len(images))
    links=[] 
    for i in range(1,len(images)): 
        try: 
            driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').click() 
            # links.append(driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img').get_attribute('src')) 
            links.append(driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img').get_attribute('src')) # 검사 copy x_path
            driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[2]/a').click() 
            print(keyword+' 링크 수집 중..... number :'+str(i)+'/'+str(len(images))) 
        except: 
            continue 

############################################# 이미지 다운로드 #############################################
        forbidden=0 
        for k,i in enumerate(links): 
            try: 
                url = i 
                start = time.time() 
                urllib.request.urlretrieve(url, "./"+keyword+"/"+keyword+"_"+str(k-forbidden)+".jpg")
                print(str(k+1)+'/'+str(len(links))+' '+keyword+' 다운로드 중....... Download time : '+str(time.time() - start)[:5]+' 초') 
            except: 
                forbidden+=1 
                continue 

    print(keyword+' ---다운로드 완료---')
    driver.close()

if __name__=='__main__': 
    pool = Pool(processes=4) # 4개의 프로세스를 사용합니다.     # Pool : 처리할 일들을 바닥에 뿌려놓고 알아서 분산처리 
    pool.map(image_download, keyword)

print('끝')
