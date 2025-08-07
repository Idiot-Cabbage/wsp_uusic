## 配置 api 服务(配置文件[web/static/config.json])
```
{
	"development": {
		"API_BASE": "http://localhost:16019/",
		"APP_ID": "test123"
	},
	"production": {
		"API_BASE": "http://localhost:16019/",
		"APP_ID": "produ123"
	}
}
```
## python  运行 http 服务
```
cd web
python -m http.server 8000
```