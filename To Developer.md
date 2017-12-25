# To Developer

​														给框架的一些说明   @author  bluebird



## 1. 准备环境

1.  推荐使用python的虚拟环境,在python3的环境下安装pip

```
pip3 install virtualenv
```

2. 将项目克隆到本地

  ```
git clone https://github.com/NJUxCloud/Cloud-Server.git
  ```

3. 在项目中创建虚拟环境，进入项目的路径，创建的目录不需要上传到github，所以如果起了不同的虚拟环境名字，要在gitignore中加注释

```
virtualenv -p python3 djangoENV
```

4. 激活虚拟环境，并安装所需要的依赖，如果使用pycharm，可以直接使用pycharm中的命令行

```
source ./djangoENV/bin/activate
pip install -r requirment.txt
```

5. 可以通过运行下面的命令来检查环境准备是否就绪

```
 python manage.py runserver
```



## 2. 创建app模块

1. 直接在命令行，创建项目中的app模块，比如app的名字是demo

```
python manage.py startapp demo
```

2. 在主目录（CloudServer) 的`setting.py`中的INSTALLED_APPS中添加`demo.apps.DemoConfig`
3. 在`model.py`中创建本app的数据模型，该模型将存储在数据库中，创建方式参见`demo/model.py`代码
4. 创建`serializers.py`在这个文件中，编写本app的serializers，serializers负责处理前端和后端的数据传输转化，但是要注意有时候不仅仅要为本模块的数据模型编写serializers，可以参考`demo/serializers.py`代码
5. 编写`view.py`，这个里面是真正处理逻辑的模块，我比较喜欢基于类的模板，所有的逻辑类都继承ApiView，然后我们为其添加`post` `get`等方法，注意这里的逻辑处理类，并不一定对应自己定义的数据模型，可以自己随便定义。可以参考`demo/views.py`。我感觉官方文档更加推荐我们使用一定的viewsets，我觉得也是可以的。如果有必要使用装饰器也可是可以的。
6. 关于`authentication`，验证的目的是为了在客户在使用一个api之前要检测它是否登录，如果登录了就会在request.user中添加用户名，这个过程restful框架已经帮我们做了很多，我们要做的就是在那些需要用户认证的逻辑类前面添加`authentication_classes = (SessionAuthentication, TokenAuthentication)`这段代码，首先`SessionAuthentication`应该是必要的，这个应该是可以使用session 的，`TokenAuthentication`是我选的，也有一些别的选择，我为什么选择这种方法是因为我使用的登录框架会创建和返回token，我就姑且认为这种认证比较方便吧。大家可以仔细看一下关于验证部分的文档。
7. 创建`permission.py`，验证用户之后只是知道了用户是否登录或者用户是谁，但是不能阻止没有权限的用户去执行API，而permission可以做到，所以在验证之后，要在`views.py`加上`permission_classes = (IsAuthenticated,)`代码（注意那个逗号是必要的）这里`IsAuthenticated`是框架自己为我们写好的，只有登录用户才可以使用这个api。我们也可以自己在`permission.py`中编写自己的permission，可以参考`demo/permission.py`代码。
8. 创建`urls.py`这里来编写，api的访问路径，因为我们使用的是django2.0，所以原来的`url`被替换成了`path`,但`path`不支持正则表达式，如果想直接使用一些框架文档或者restful框架的代码,可以使用`re_path`这个应该才是原来`url`的替代品。也要记得在主目录的`urls.py`中添加只想本app的url
9. 最后要在主目录下`setting.py`中添加自己引用一些框架和安装的app，如果使用了一些第三方依赖，要记得在`requirment.txt`中添加，方便别的开发者安装



## 3. 运行和检测

1. 数据库的迁移，迁移的目的是为了让数据库，自己创建你再`model.py`创建的数据模型，要创键app的迁移文件，然后进行迁移

```python
python manage.py makemigrations demo 	# 创建demo app的数据迁移文件
python manage.py migrate    			# 迁移所有创建模型到数据库
```

2. 运行server

```
python manage.py runserver  
```

3. 检测api

   1. 首先可以直接在浏览器中，输入你再`urls.py`中自己定义的api，比如登录`127.0.0.1:8000/rest-auth/login`就进入了登录的api，但是这个api需要输入参数，我们可以再框架给的interface里面填写。要注意的是，这种方法看起来很方便，特别是登录和验证这一块，以为我们用的第三方框架给了我们一个interface，如果使我们自己编写的，就需要自己把文件编写成一个json格式，这就有点麻烦了
   2. 使用httpie，这个工具我已经加到`requirment.txt`,所以我们可以直接使用。比如我们可以直接传递表单数据，来验证登录，但是对于那些需要验证的api来说，我们要稍微麻烦一点，要加入用户的token

   ```python
   http --form POST http://127.0.0.1:8000/rest-auth/login/ username='admin' password='passw123' email='151250145@smail.nju.edu.cn' #检测是否可以登录
   ```

   ```python
   https --form GET http://127.0.0.1:8000/demo/ 'Authorization: Token ef0d45b7dd416cec4d113bae0766bece47528b54'  # 以这个Token的身份，查看demo中bills的数据
   ```


4. 对登录的说明，我们无需自己书写登录的api，因为这个框架已经写好了。我对于登录和注册采用`django-rest-auth`框架，我们可以直接在浏览器使用这些登录和注册，当然这里面提供的interface是给我们开发人员的。前端的登录注册界面，可以通过框架给出的api来运行，这些api，在http://django-rest-auth.readthedocs.io/en/latest/api_endpoints.html给出了说明



## 4.未完待续

1. 单元测试
2. 集成测试
3. swagger文档