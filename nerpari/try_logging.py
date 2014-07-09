import logging

class MyLogger():
    def __init__(self, console_log_level=logging.DEBUG):
        #logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        #handler that writes everything to file
        h1 = logging.FileHandler('crawler.log')
        h1.setLevel(logging.INFO)
        self.logger.addHandler(h1)

        #handler that writes to console at a level as specified
        h2 = logging.StreamHandler()
        h2.setLevel(console_log_level)
        self.logger.addHandler(h2)

    def cause_exception(self):
        self.logger.exception('This is an exception!')

    def print_info(self):
        self.logger.info('This is information.')

    def print_debug(self):
        self.logger.debug('This is debug information.')