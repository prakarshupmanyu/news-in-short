import scrapy

MAX_PAGE_NO = 64880

urls = []

class QuotesSpider(scrapy.Spider):
    name = "lemonde-fr-spider"

    def start_requests(self):
        urls = []
        for pageno in range(13192,MAX_PAGE_NO+1):
                url = 'http://www.lemonde.fr/recherche/?keywords=+politique&page_num={}'.format(int(pageno))
                urls.append(url)

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse1)

    def parse1(self, response):

        search_string = "politique"
        filename = "/home/sarthak/Mydata/Projects/silicon-beach-data/urls/" + "lemonde-fr-Spider-links_" + search_string+  ".csv"
        with open(filename, 'a') as f:
            try:
                div = response.xpath('//h3')
                for links in div.xpath('.//a[contains(@href, ".html")]/@href'):
                    linkList = links.extract()
                    current_link = "http://www.lemonde.fr/" + linkList + "\n"
                    print(current_link)
                    f.write(current_link)
                    urls.append(current_link)
            except Exception as e:
                print("sorry the link could not be extracted : ", e)
        self.log('Saved file %s' % filename)
        print("######################\n\n\n  ", len(urls))

