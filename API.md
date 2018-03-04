<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<!-- sudo npm install -g doctoc-->
<!-- doctoc ./API.md -->

- [API文档](#api%E6%96%87%E6%A1%A3)
  - [注](#%E6%B3%A8)
  - [用户模块](#%E7%94%A8%E6%88%B7%E6%A8%A1%E5%9D%97)
    - [`POST`  登录 /rest-auth/login/](#post--%E7%99%BB%E5%BD%95-rest-authlogin)
    - [`POST` 重置密码 /rest-auth/password/reset/](#post-%E9%87%8D%E7%BD%AE%E5%AF%86%E7%A0%81-rest-authpasswordreset)
    - [`POST` 修改密码 /rest-auth/password/change/](#post-%E4%BF%AE%E6%94%B9%E5%AF%86%E7%A0%81-rest-authpasswordchange)
    - [`AUTH` `GET` `PUT` `PATCH` 用户信息 /rest-auth/user/](#auth-get-put-patch-%E7%94%A8%E6%88%B7%E4%BF%A1%E6%81%AF-rest-authuser)
    - [`POST` 注册 /rest-auth/registration/](#post-%E6%B3%A8%E5%86%8C-rest-authregistration)
    - [`POST` 注册时验证邮箱 /rest-auth/registration/verify-email/](#post-%E6%B3%A8%E5%86%8C%E6%97%B6%E9%AA%8C%E8%AF%81%E9%82%AE%E7%AE%B1-rest-authregistrationverify-email)
  - [数据管理模块（包含了数据上传和代码上传）](#%E6%95%B0%E6%8D%AE%E7%AE%A1%E7%90%86%E6%A8%A1%E5%9D%97%E5%8C%85%E5%90%AB%E4%BA%86%E6%95%B0%E6%8D%AE%E4%B8%8A%E4%BC%A0%E5%92%8C%E4%BB%A3%E7%A0%81%E4%B8%8A%E4%BC%A0)
    - [`AUTH` `POST` 创建新模型 /data/create/]
    - [`AUTH` `GET` 获得用户已上传数据列表 /data/list/](#auth-get-%E8%8E%B7%E5%BE%97%E7%94%A8%E6%88%B7%E5%B7%B2%E4%B8%8A%E4%BC%A0%E6%95%B0%E6%8D%AE%E5%88%97%E8%A1%A8-datalist)
    - [`AUTH` `GET` 根据数据id获得数据内容 /data/([0-9]+)/](#auth-get-%E6%A0%B9%E6%8D%AE%E6%95%B0%E6%8D%AEid%E8%8E%B7%E5%BE%97%E6%95%B0%E6%8D%AE%E5%86%85%E5%AE%B9-data0-9)
    - [`AUTH` `POST` 用户上传文件 /data/](#auth-post-%E7%94%A8%E6%88%B7%E4%B8%8A%E4%BC%A0%E6%96%87%E4%BB%B6-data)
    - [`AUTH` `DELETE` 删除数据文件 /data/([0-9]+)/](#auth-delete-%E5%88%A0%E9%99%A4%E6%95%B0%E6%8D%AE%E6%96%87%E4%BB%B6-data0-9)
  - [数据预处理](#%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86)
    - [`AUTH` `GET` 执行数据预处理操作 /preprocess/operations/list/](#auth-get-%E8%8E%B7%E5%BE%97%E6%89%80%E6%9C%89%E6%93%8D%E4%BD%9C%E5%88%97%E8%A1%A8-preprocessoperationslist)
  - [参数及运行模块](#%E5%8F%82%E6%95%B0%E5%8F%8A%E8%BF%90%E8%A1%8C%E6%A8%A1%E5%9D%97)
    - [`AUTH` `GET` 获得算法列表及参数信息 /generation/options/list/](#auth-get-%E8%8E%B7%E5%BE%97%E7%AE%97%E6%B3%95%E5%88%97%E8%A1%A8%E5%8F%8A%E5%8F%82%E6%95%B0%E4%BF%A1%E6%81%AF-generationoptionslist)
    - [`AUTH` `GET` 获得下一步可选的算法列表及参数信息 /generation/options/next/](#auth-get-%E8%8E%B7%E5%BE%97%E4%B8%8B%E4%B8%80%E6%AD%A5%E5%8F%AF%E9%80%89%E7%9A%84%E7%AE%97%E6%B3%95%E5%88%97%E8%A1%A8%E5%8F%8A%E5%8F%82%E6%95%B0%E4%BF%A1%E6%81%AF-generationoptionsnext)
    - [`AUTH` `POST` 生成代码 /generation/generate/](#auth-post-%E7%94%9F%E6%88%90%E4%BB%A3%E7%A0%81-generationgenerate)
    - [`AUTH` `POST` 获得基本结果 /generation/run/basic/](#auth-post-%E8%8E%B7%E5%BE%97%E5%9F%BA%E6%9C%AC%E7%BB%93%E6%9E%9C-generationrunbasic)
    - [`AUTH` `GET` 获得详细结果 /generation/run/details/](#auth-get-%E8%8E%B7%E5%BE%97%E8%AF%A6%E7%BB%86%E7%BB%93%E6%9E%9C-generationrundetails)
    - [`AUTH` `GET` 运行，获得运行实时数据 /generation/run/runtime/](#auth-get-%E8%BF%90%E8%A1%8C%E8%8E%B7%E5%BE%97%E8%BF%90%E8%A1%8C%E5%AE%9E%E6%97%B6%E6%95%B0%E6%8D%AE-generationrunruntime)
    - [`AUTH` `GET` 重新加载模型参数 /generation/restore/([0-9]+)/](#auth-get-%E9%87%8D%E6%96%B0%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0-generationrestore0-9)
    - [`AUTH` `POST` 停止模型运行 /generation/run/stop/](#auth-post-%E5%81%9C%E6%AD%A2%E6%A8%A1%E5%9E%8B%E8%BF%90%E8%A1%8C-generationrunstop)
    - [`AUTH` `POST` 暂停模型运行 /generation/run/pause/](#auth-post-%E6%9A%82%E5%81%9C%E6%A8%A1%E5%9E%8B%E8%BF%90%E8%A1%8C-generationrunpause)
  - [模型模块](#%E6%A8%A1%E5%9E%8B%E6%A8%A1%E5%9D%97)
    - [`AUTH` `GET` 获得用户模型列表 /models/list/](#auth-get-%E8%8E%B7%E5%BE%97%E7%94%A8%E6%88%B7%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8-modelslist)
    - [`AUTH` `GET` 获得模型详情 /models/([0-9]+)/](#auth-get-%E8%8E%B7%E5%BE%97%E6%A8%A1%E5%9E%8B%E8%AF%A6%E6%83%85-models0-9)
    - [`AUTH` `DELETE` 删除模型 /models/([0-9]+)/](#auth-delete-%E5%88%A0%E9%99%A4%E6%A8%A1%E5%9E%8B-models0-9)
    - [`AUTH` `GET` 比较模型 /models/compare/](#auth-get-%E6%AF%94%E8%BE%83%E6%A8%A1%E5%9E%8B-modelscompare)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



## API文档
### 注
文档中含有`AUTH`标签的，需要在header中添加

>  Authorization : Token djjhffkjhfsfsnfwu43920dnfnjshks
>  Content-Type: application/json
 
### 用户模块
#### `POST`  登录 /rest-auth/login/ 

- username
- email
- password

```json
{
    "key": "ec38605dc9111748d670ebf09edae8074014bede"
}
```
```json
{
    "non_field_errors": [
        "Unable to log in with provided credentials."
    ]
}
```

#### `POST` 重置密码 /rest-auth/password/reset/
将发送邮件至用户注册邮箱
**TODO** 如果该邮箱未被注册，返回的json和正常情况下一样，但用户不会收到邮件。

```json
{
    "detail": "Password reset e-mail has been sent."
}
```


####`POST` 确认重置密码 /rest-auth/password/reset/confirm/

- uid
- token
- new_password1
- new_password2

`uid`和`token`在email的链接中给出


#### `POST` 修改密码 /rest-auth/password/change/

- new_password1
- new_password2
- old_password


#### `AUTH` `GET` `PUT` `PATCH` 用户信息 /rest-auth/user/

- username

```json
{
    "pk": 3,
    "username": "zqh",
    "email": "151250206@smail.nju.edu.cn",
    "first_name": "",
    "last_name": ""
}
```


#### `POST` 注册 /rest-auth/registration/

- username
- password1
- password2
- email

```json
{
    "non_field_errors": [
        "The two password fields didn't match."
    ],
    "username": [
        "A user with that username already exists."
    ],
    "password1": [
        "This password is too short. It must contain at least 8 characters.",
        "This password is too common.",
        "This password is entirely numeric."
    ],
    "email": [
        "A user is already registered with this e-mail address."
    ]
}
```

#### `POST` 注册时验证邮箱 /rest-auth/registration/verify-email/

- key


-------

### 数据管理模块（包含了数据上传和代码上传）
#### `AUTH` `POST` 创建新模型 /data/create/
   
- modelName

#### '`AUTH` `POST` 创建新模型 /data/tag/

- modelName
- file(格式为json)

#### `AUTH` `GET` 获得用户已上传数据列表 /data/list/
```json
[
    {
        "id": 8,
        "created_at": "2018-01-09T06:24:27.185871Z",
        "file_type": "picture",
        "file_name": "DBSCAN算法过程_20180109062425.png",
        "owner": 3
    },
    {
        "id": 9,
        "created_at": "2018-01-09T06:38:27.502036Z",
        "file_type": "audio",
        "file_name": "RadioInMyHead_20180109063825.mp3",
        "owner": 3
    },
    {
        "id": 10,
        "created_at": "2018-01-09T06:43:14.721760Z",
        "file_type": "code",
        "file_name": "nn_binary_classification_20180109064313.py",
        "owner": 3
    }
]
```

#### `AUTH` `GET` 根据数据id获得数据内容 /data/([0-9]+)/
```text
params: {
    # 文件夹目录结构中会使用到
    relative_path: myfile/nn/nn_binary_classification.py
}
```
return：
图片文件、音频文件、python文件直接返回
目录结构返回为

```json
{
    "image_process.py": "image_process.py", 
    "mnist": {
        "mnist_inference.py": "mnist_inference.py", 
        "mnist_train.py": "mnist_train.py", 
        "mnist_eval.py": "mnist_eval.py", 
        "__init__.py": "__init__.py", 
        "__pycache__": {
            "mnist_train.cpython-36.pyc": "mnist_train.cpython-36.pyc", 
            "mnist_inference.cpython-36.pyc": "mnist_inference.cpython-36.pyc", 
            "__init__.cpython-36.pyc": "__init__.cpython-36.pyc"
        }
    }
}
```
csv文件返回为

```json
 [
    {
        "id": "1", 
        "name": "Java", 
        "add_time": "2017-12-16", 
        "num": "2", 
        "price": "9", 
        "owner": "false"
    }, 
    {
        "id": "2", 
        "name": "Python", 
        "add_time": "2017-12-16", 
        "num": "3", 
        "price": "9.8", 
        "owner": "false"
    }
]

```

#### `AUTH` `POST` 用户上传文件 /data/list/
支持上传url链接、压缩包、单个文件

- file_type: url / single / zip
- file_class: doc / code / audio / picture
- file
- url

返回
```json
{
  "data_id":1
}
```

#### `AUTH` `DELETE` 删除数据文件 /data/(0-9]+)/
```json
{
    "message": "success"
}
```
如果不存在对应id的数据文件
返回状态为 HTTP_204_NO_CONTENT

-------

### 数据预处理
#### `AUTH` `POST` 执行数据预处理操作 /preprocess/
POST 传入的json格式:

```json
{
    "dataId":1,
    "operations":[
      {
        "operationName":"上下翻转",
        "overlap":true,
        "value1":null, // 即使不需要输入数值也加入这一项，为null
        "value2":null
      },
      {
        "operationName":"亮度调整",
        "overlap":false,
        "value1":0.4,
        "value2":null
      },
      {
        "operationName":"中值滤波",
        "overlap":false,
        "value1":5,
        "value2":5
      }
    ]
}
```
传的json中的operations要求有序，不同的操作顺序会带来不同的结果

返回500或200

-------

### 参数及运行模块
#### `AUTH` `GET` 获得算法列表及参数信息 /generation/options/list/
#### `AUTH` `GET` 获得下一步可选的算法列表及参数信息 /generation/options/next/
#### `AUTH` `POST` 生成代码 /generation/generate/
#### `AUTH` `POST` 获得基本结果 /generation/run/basic/
#### `AUTH` `GET` 重新加载模型参数 /generation/restore/([0-9]+)/
#### `AUTH` `POST` 停止模型运行 /generation/run/stop/
#### `AUTH` `POST` 暂停模型运行 /generation/run/pause/

-------

### 结果部分
#### `AUTH` `GET` 获得kubernetes结果 /runtime/kubernetes/
获取的结果格式
```json
{
    "Addresses": {
        "Hostname": "k8s-node-3",
        "InternalIP": "119.23.51.139"
    },
    "Conditions": [
        {
            "LastTransitionTime": "Mon, 26 Feb 2018 17:49:10 +0800",
            "Message": "kubelet has sufficient disk space available",
            "Status": "False",
            "LastHeartbeatTime": "Sun, 04 Mar 2018 19:49:39 +0800",
            "Type": "OutOfDisk",
            "Reason": "KubeletHasSufficientDisk"
        },
        {
            "LastTransitionTime": "Mon, 26 Feb 2018 17:49:10 +0800",
            "Message": "kubelet has sufficient memory available",
            "Status": "False",
            "LastHeartbeatTime": "Sun, 04 Mar 2018 19:49:39 +0800",
            "Type": "MemoryPressure",
            "Reason": "KubeletHasSufficientMemory"
        },
        {
            "LastTransitionTime": "Mon, 26 Feb 2018 17:49:10 +0800",
            "Message": "kubelet has no disk pressure",
            "Status": "False",
            "LastHeartbeatTime": "Sun, 04 Mar 2018 19:49:39 +0800",
            "Type": "DiskPressure",
            "Reason": "KubeletHasNoDiskPressure"
        },
        {
            "LastTransitionTime": "Mon, 26 Feb 2018 17:49:10 +0800",
            "Message": "kubelet is posting ready status",
            "Status": "True",
            "LastHeartbeatTime": "Sun, 04 Mar 2018 19:49:39 +0800",
            "Type": "Ready",
            "Reason": "KubeletReady"
        }
    ],
    "Non-terminated Pods": [
        {
            "Memory Requests": "0 (0%)",
            "CPU Limits": "0 (0%)",
            "Namespace": "default",
            "Memory Limits": "0 (0%)",
            "CPU Requests": "0 (0%)",
            "Name": "tensorflow-ps-rc-tmrww"
        },
        {
            "Memory Requests": "0 (0%)",
            "CPU Limits": "0 (0%)",
            "Namespace": "default",
            "Memory Limits": "0 (0%)",
            "CPU Requests": "0 (0%)",
            "Name": "tensorflow-worker-rc-j9ptp"
        },
        {
            "Memory Requests": "0 (0%)",
            "CPU Limits": "0 (0%)",
            "Namespace": "default",
            "Memory Limits": "0 (0%)",
            "CPU Requests": "0 (0%)",
            "Name": "tensorflow-worker-rc-q67pm"
        },
        {
            "Memory Requests": "0 (0%)",
            "CPU Limits": "0 (0%)",
            "Namespace": "kube-system",
            "Memory Limits": "0 (0%)",
            "CPU Requests": "0 (0%)",
            "Name": "kube-flannel-ds-dgvf8"
        },
        {
            "Memory Requests": "0 (0%)",
            "CPU Limits": "0 (0%)",
            "Namespace": "kube-system",
            "Memory Limits": "0 (0%)",
            "CPU Requests": "0 (0%)",
            "Name": "kube-proxy-dt8hx"
        }
    ],
    "Capacity": {
        "cpu": "1",
        "memory": "1883724Ki",
        "pods": "110"
    },
    "Allocatable": {
        "cpu": "1",
        "memory": "1781324Ki",
        "pods": "110"
    },
    "System Info": {
        "Kube-Proxy Version": "v1.8.2",
        "Architecture": "amd64",
        "Container Runtime Version": "docker://Unknown",
        "Operating System": "linux",
        "OS Image": "CentOS Linux 7 (Core)",
        "Kubelet Version": "v1.8.2"
    }
}
```

#### `AUTH` `GET` 获得tensorflow的结果 /runtime/train/{modelname}/
modelname为该训练模型的名称
获取的结果格式
```json
{
    "every_result": [
        {
            "accuracy": "0.080000",
            "duration": "0.000000",
            "step": "0"
        },
        {
            "accuracy": "0.240000",
            "duration": "0.138934",
            "step": "100"
        },
        {
            "accuracy": "0.240000",
            "duration": "0.141843",
            "step": "200"
        },
        {
            "accuracy": "0.400000",
            "duration": "0.139994",
            "step": "300"
        },
        {
            "accuracy": "0.360000",
            "duration": "0.138250",
            "step": "400"
        },
        {
            "accuracy": "0.580000",
            "duration": "0.145144",
            "step": "500"
        },
        {
            "accuracy": "0.540000",
            "duration": "0.143614",
            "step": "600"
        },
        {
            "accuracy": "0.540000",
            "duration": "0.142053",
            "step": "700"
        },
        {
            "accuracy": "0.660000",
            "duration": "0.141187",
            "step": "800"
        },
        {
            "accuracy": "0.640000",
            "duration": "0.151048",
            "step": "900"
        },
        {
            "accuracy": "0.560000",
            "duration": "0.138814",
            "step": "1000"
        }
    ],
    "final_accuracy": 0.44
}

```


-------

### 模型模块
#### `AUTH` `GET` 获得用户模型列表 /construct/config/
获取的结果格式
```json
["modelname2","modelname"]
```

#### `AUTH` `GET` 获得模型详情 /construct/detail/{modelname}/
获取的结果格式
```json
{"iter":1000,
 "learning_rate":0.01,
 "ratio":0.8,
 "loss_name":"entropy",
 "optimizer_name":"GradientDescentOptimizer",
 "net_type":"CNN",
 "net_config":{
    "middle_layer":[{
        "layer":"conv",
        "filter":[2,2,10]
        },{
        "layer":"conv",
        "filter":[2,2,20]
        },{
        "layer":"pool"
        },{
        "layer":"norm"
        },{
        "layer":"active"
        },{
        "layer":"connect"
        },{
        "layer":"connect"
    }],
 "output_layer":{}
 }
}

```

#### `AUTH` `GET` 使用模型 /construct/detail/{modelname}/
获取的结果格式
```json
{
  "result":"success",
  "message":1
 }
```







