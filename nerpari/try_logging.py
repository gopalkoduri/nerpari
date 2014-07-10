import logging

class MyLogger():
    def __init__(self, console_log_level=logging.DEBUG):
        #logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        #handler that writes everything to file
        h1 = logging.FileHandler('crawler.log')
        h1.setLevel(logging.WARN)
        self.logger.addHandler(h1)

        #handler that writes to console at a level as specified
        h2 = logging.StreamHandler()
        h2.setLevel(console_log_level)
        self.logger.addHandler(h2)

    def log(self):
        self.logger.debug('This is debug information.')
        self.logger.info('This is just plain information.')
        self.logger.warning('This is a warning.')
        self.logger.error('This is an error!')
        self.logger.critical('Things got critical!')