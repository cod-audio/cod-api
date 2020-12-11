heroku buildpacks:set heroku/python
heroku buildpacks:add --index 1 heroku-community/apt

web: gunicorn codapi:app