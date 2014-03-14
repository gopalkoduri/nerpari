from celery import Celery

app = Celery('text',
             broker='amqp://guest:guest@devaraya.s.upf.edu',
             #backend='amqp://',
             include=['tasks'])

# Optional configuration, see the application user guide.
app.conf.update(
    CELERYD_CONCURRENCY=2,
    CELERYD_POOL_RESTARTS=True,
)

if __name__ == '__main__':
    app.start()