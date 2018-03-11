# Cloud-Server
## 概述
NJUCloud在线深度学习平台的服务端，对外提供的RESTful API功能包括：
- 用户管理
- 数据上传
- 数据预处理
- 模型构建
- 实时监测
- 结果展示
- 手写数字识别

该项目利用Django框架搭建，使用TensorFlow构建深度学习模型，利用Kubernetes搭建分布式计算环境

## 本地环境搭建
1. 推荐使用python的虚拟环境,在python3的环境下安装pip
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

4. 激活虚拟环境，并安装所需要的依赖，如果使用PyCharm，可以直接使用PyCharm中的命令行

```
source ./djangoENV/bin/activate
pip install -r requirment.txt
```

5. 可以通过运行下面的命令来检查环境准备是否就绪

```
python3 manage.py runserver
```

6. 配置数据库：在CloudServer/settings.py中修改`DATABASES`为本地数据库信息
```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'CloudDB',
        'USER': 'abcde',
        'PASSWORD': 'abcde',
        'HOST': 'xxx.xx.xxx.xx',
        'PORT': '3306',
    }
}
```

7. 进行数据迁移，迁移所有创建模型到数据库
```python
python manage.py migrate
```

8. 修改本地数据存储地址：在CloudServer/global_settings.py中修改`LOCAL_STORAGE_PATH`为本地数据存储地址
```
LOCAL_STORAGE_PATH = '/Users/myname/Downloads/'
```

9. 运行项目
```
python manage.py runserver  
```

10. API测试
使用httpie，这个工具已经加到`requirment.txt`,可以直接使用。如可以直接传递表单数据，来验证登录，但是对于那些需要验证的api来说，要加入用户的token
```python
http --form POST http://127.0.0.1:8000/rest-auth/login/ username='admin' password='passw123' email='151250145@smail.nju.edu.cn' #检测是否可以登录
```

```python
https --form GET http://127.0.0.1:8000/demo/ 'Authorization: Token ef0d45b7dd416cec4d113bae0766bece47528b54'  # 以这个Token的身份，查看demo中bills的数据
```

或使用Postman进行API测试

## 第三方库
- mysqlclient
- djangorestframework
- markdown
- django-filter
- pygments
- httpie
- django-rest-auth==0.9.2
- django-allauth>=0.24.1
- six==1.9.0
- django-rest-swagger==2.0.7
- tensorflow
- paramiko
- NumPy>= 1.8.2
- opencv-python
- SciPy>= 0.13.3
- scikit-learn
- pillow
- pandas
- django-cors-middleware
