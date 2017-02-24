import scrapy

MAX_PAGE_NO = 13186

urls = []

class QuotesSpider(scrapy.Spider):
    name = "ouest-france-spider"

    def start_requests(self):
        urls = []
        for pageno in range(10000,MAX_PAGE_NO):
                url = 'http://www.ouest-france.fr/search/site/politique/?page={}'.format(int(pageno))
                urls.append(url)

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse1)

    def parse1(self, response):

        search_string = "politique"
        filename = "/home/melvin/Documents/USC/news-in-short-data/urls/" + "ouest-france-spider-links_" + search_string+  ".csv"
        with open(filename, 'a') as f:
            try:
                div = response.xpath('//h2[@class="title"]')
                for links in div.xpath('a[contains(@href,".fr")]/@href'):
                    linkList = links.extract()
                    current_link = linkList + "\n"
                    #print(current_link)
                    f.write(current_link)
                    #urls.append(current_link)
            except Exception as e:
                print("sorry the link could not be extracted : ", e)
        #self.log('Saved file %s' % filename)
        #print("######################\n\n\n  ", len(urls))

