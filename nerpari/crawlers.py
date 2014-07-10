from __future__ import unicode_literals
from BeautifulSoup import BeautifulSoup, Tag
import cPickle
import logging
import requests
import re
import time
import uuid
import sys

class Crawler():
    def __init__(self, storage_dir, log_level=logging.DEBUG):
        self.storage_dir = storage_dir

        self.crawl_state = {"visited_urls": set()}
        self.persistence_interval = 20
        self.read_count = 0

        self.crawl_interval = 0.25 #in seconds
        self.prev_crawl_timestamp = time.time()

        #logging
        self.logger = logging.getLogger('crawler')
        self.logger.setLevel(logging.DEBUG)

        #handler that writes everything to file
        h1 = logging.FileHandler(self.storage_dir + '/crawler.log')
        h1.setLevel(log_level)
        self.logger.addHandler(h1)

        #handler that writes to console at a level as specified
        h2 = logging.StreamHandler()
        h2.setLevel(log_level)
        self.logger.addHandler(h2)

    def store_state(self):
        self.logger.info('Storing crawl state. #Visted URLs: %d', len(self.crawl_state['visited_urls']))
        cPickle.dump(self.crawl_state, file(self.storage_dir+'/crawl_state.pickle', 'w'))

    def load_state(self):
        try:
            self.crawl_state = cPickle.load(file(self.storage_dir+'/crawl_state.pickle'))
        except IOError:
            self.logger.error("Can't read the crawl_state file!")
            sys.exit()

    def read(self, url):
        #compare the read count with persistence interval
        if self.read_count >= self.persistence_interval:
            self.store_state()
            self.read_count = 0

        #control the crawl rate
        cur_timestamp = time.time()
        if cur_timestamp - self.prev_crawl_timestamp < self.crawl_interval:
            time.sleep(cur_timestamp - self.prev_crawl_timestamp)
        self.prev_crawl_timestamp = cur_timestamp

        #get the data and send!
        self.logger.info('Getting data from: %s', url)
        data = requests.get(url)
        self.read_count += 1

        return data.text


class TheHindu(Crawler):
    def __init__(self, storage_dir, log_level=logging.INFO):
        Crawler.__init__(self, storage_dir, log_level)
        self.home = 'http://hindu.com/thehindu/fr/arcfr.htm'
        self.current_data = {'headline': '',
                             'news': '',
                             'date': '',
                             'photo': {'caption':'', 'url':''},
                             'url': ''}

        self.date_pattern = '(fr/\d\d\d\d/\d\d/\d\d)'
        self.crawl_state['top_level_urls'] = set()
        self.crawl_state['city_edn_urls'] = set()
        self.crawl_state['article_urls'] = set()

    def get_toplevel_urls(self):
        """
        From the hindu archive home page, this function gets urls corresponding to all dates.
        :return: nothing, but modifies the crawl state adding more urls.
        """
        data = self.read(self.home)
        soup = BeautifulSoup(data)
        temp = soup.findAll('a')

        for url in temp:
            url_data = dict(url.attrs)
            url = url_data['href']
            self.logger.debug('Checking '+url)
            res = re.search(self.date_pattern, url)
            if res:
                self.crawl_state['top_level_urls'].add('http://www.hindu.com'+url)

        self.crawl_state['visited_urls'].add(self.home)

    def get_city_edn_urls(self, top_level_url):
        """
        Given a top level url, this function gets all urls for all the city editions for that date.
        :param top_level_url: a top level url from the archive (i.e., for a date).
        :return: nothing, but modifies the crawl state adding more urls.
        """
        data = self.read(top_level_url)
        soup = BeautifulSoup(data)
        temp = soup.findAll('a')

        for city_edn_url in temp:
            url_data = dict(city_edn_url.attrs)
            try:
                city_edn_url = url_data['href']
            except KeyError:
                continue
            self.logger.debug('Checking '+city_edn_url)
            res = re.search(self.date_pattern, city_edn_url)
            if res:
                self.crawl_state['city_edn_urls'].add('http://www.hindu.com'+city_edn_url)

        self.crawl_state['visited_urls'].add(top_level_url)

    def get_music_article_urls(self, city_edn_url):
        """
        From a given city edition url, this function gets links to all the music articles in it.
        :param city_edn_url: a city edition url.
        :return: nothing, but modifies the crawl state adding more urls.
        """
        data = self.read(city_edn_url)
        soup = BeautifulSoup(data)

        temp = soup.findAll('font', attrs={'class': 'sectionhead'})
        music_tag = None
        for t in temp:
            if t.text.lower() == 'music':
                music_tag = t

        if not music_tag:
            self.crawl_state['visited_urls'].add(city_edn_url)
            return

        next_tag = music_tag.findNext('font', attrs={'class': 'sectionhead'})
        if not next_tag:
            next_tag = music_tag.findNext('font', attrs={'color': 'red'})

        s1 = music_tag.findAllNext('a')
        s2 = next_tag.findAllPrevious('a')

        date_pattern = re.search(self.date_pattern, city_edn_url).group()

        for article_url in set(s1).intersection(s2):
            url_data = dict(article_url.attrs)
            self.crawl_state['article_urls'].add('http://www.hindu.com/thehindu/'+date_pattern+'/'+url_data['href'])

        self.crawl_state['visited_urls'].add(city_edn_url)

    def get_contents(self, url):
        """
        Given an article's url, it returns the
        :param url: URL pointing to an article.
        :return: nothing, modifies the crawl_state adding/changing article_urls/visited_urls.
        """
        #Author is messed up! discarded.

        #Date can be taken from URL
        #Headline: ('font', attrs={'class': 'storyhead'})
        #Media is all in the center tag after the headline starts
        #News is whatever text can be found in p tag after headline and before ('font', attrs={'class': 'leftnavi', 'face': 'verdana'})

        #just to make sure this is clean!
        self.current_data = {'headline': '',
                             'news': '',
                             'date': '',
                             'photo': {'caption':'', 'url':''},
                             'url': ''}

        data = self.read(url)
        soup = BeautifulSoup(data)

        self.current_data['url'] = url
        date_pattern = re.search(self.date_pattern, url).group()
        self.current_data['date'] = date_pattern[3:]

        head = soup.find('font', attrs={'class': 'storyhead'})
        self.current_data['headline'] = head.text

        end = soup.findAll('font', attrs={'class': 'leftnavi', 'face': 'verdana'})
        s1 = head.findAllNext('p')
        s2 = end[0].findAllPrevious('p')

        #can't take an set intersection as it screws up order
        common = []
        for p in s1:
            if p in s2:
                common.append(p)

        for p in common[:-2]:
            media = False
            children = list(p.childGenerator())
            for child in children:
                if type(child) != Tag:
                    continue
                if child.name == 'center':
                    media = True
                    try:
                        img_data = dict(child.img.attrs)
                    except AttributeError:
                        self.logger.warning('%s is not an expected center tag.', child)
                        continue
                    try:
                        self.current_data['photo']['url'] = 'http://www.hindu.com/' + date_pattern + '/' + img_data['src'][3:]
                        self.current_data['photo']['caption'] = child.b.text
                    except:
                        self.logger.warning('Media in %s does not seem to have all the information', url)
                    break
            if media:
                continue
            else:
                script = p.findChild('script')
                if not script:
                    self.current_data['news'] += p.text

        self.crawl_state['visited_urls'].add(url)

    def boot(self, resume=True):
        """
        Starts afresh the crawl, unless asked to resume.
        :param resume: If True, tries to read the crawl state and resumes. Otherwise, starts afresh.
        :return: None, periodically writes crawl_state to the disk, and dumps new articles crawled to storage_dir.
        """
        if resume:
            self.load_state()
        else:
            self.crawl_state['visited_urls'] = set()

        #Get top level URLs
        self.logger.info('Getting the top level URLs ...')
        self.get_toplevel_urls()

        #Get city edition URLs
        self.logger.info('Getting the city edition URLs ...')
        for url in self.crawl_state['top_level_urls']:
            if url not in self.crawl_state['visited_urls']:
                self.get_city_edn_urls(url)
        self.store_state()

        #Get music article URLs
        self.logger.info('Getting the music article URLs ...')
        for url in self.crawl_state['city_edn_urls']:
            if url not in self.crawl_state['visited_urls']:
                self.get_music_article_urls(url)
        self.store_state()

        #Get contents!
        self.logger.info('Getting the contents ...')
        for url in self.crawl_state['article_urls']:
            if url not in self.crawl_state['visited_urls']:
                self.get_contents(url)
                f_name = uuid.uuid5(uuid.NAMESPACE_URL, url.encode('utf-8')).hex
                cPickle.dump(self.current_data, file(self.storage_dir + '/' + f_name, 'w'))
        self.store_state()