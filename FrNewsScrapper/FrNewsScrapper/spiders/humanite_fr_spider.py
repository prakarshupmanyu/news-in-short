import scrapy

MAX_PAGE_NO = 4931

urls = []

class QuotesSpider(scrapy.Spider):
    name = "humanite-fr-spider"

    def start_requests(self):
        urls = []
        for pageno in range(1, MAX_PAGE_NO+1):
                url = 'http://www.humanite.fr/politique?page={}'.format(int(pageno))
                urls.append(url)

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse1)

    def parse1(self, response):

        search_string = "politique"
        filename = "/Users/TMK/git/news-in-short/FrNewsScrapper/humanite-fr-Spider-links_" + search_string + ".csv"
        with open(filename, 'a') as f:
            try:
                div = response.xpath('//div[@id="content"]//h2')
                for links in div.xpath('.//a[contains(@href, "")]/@href'):
                    linkList = links.extract()
                    current_link = "http://www.humanite.fr" + linkList + "\n"
                    f.write(current_link)
                    urls.append(current_link)
            except Exception as e:
                print("sorry the link could not be extracted : ", e)
                
        self.log('Saved file %s' % filename)
        print("######################\n\n\n  ", len(urls))

