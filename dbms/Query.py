"""
    @Author: Wilton Beltré
    @License: MIT
"""


class Query(object):
    def __init__(self, SQL_FILE):
        with open(SQL_FILE) as q:
            self.data = q.read().splitlines()

        last_key = None
        query_collect = []
        body = dict()
        for dt in self.data:
            if '--' in dt:
                last_key = dt.replace('--', '')
                query_collect = []
            else:
                query_collect.append(dt)
                body[last_key] = query_collect

        for b in body.keys():
            setattr(self, b, ' '.join(body[b]))
