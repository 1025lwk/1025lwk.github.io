---
title: Unity游戏控制基础脚本
date: 2023/2/1 13:52:41
description: 最基础的unity游戏制作，对游戏物体进行控制的脚本。

categories: Gameproduction
cover: https://cdn.wallpapersafari.com/55/68/ZOeNnY.jpg
---



## 首先声明！！！

---
* 1.脚本为本人总结，如有使用注明出处
* 2.Unity采用C#编程语言编写脚本。
* 3.脚本内有注释。

---



## 一、TimeTest（时间控制测试）


```c#
using JetBrains.Annotations;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

public class TimeTest : MonoBehaviour
{
    public GameObject Prefab;

    //计时器
    float timer = 0;

    int t = 0;

    // Start is called before the first frame update
    void Start()
    {
        //游戏开始到现在所花的时间
        Debug.Log(Time.time);

        //时间缩放值
        Debug.Log(Time.timeScale);

        //固定时间间隔
        Debug.Log(Time.fixedDeltaTime);
    }

    GameObject p;
    // Update is called once per frame
    void Update()
    {
        timer = timer + Time.deltaTime;
        //每一帧所有的时间
        //Debug.Log(Time.deltaTime);

        if (timer > 3&& t == 0) {
            Debug.Log("敌人出现了！");
            p = Instantiate(Prefab, Vector3.one, Quaternion.identity);
            t = 1;
        }

        if (timer > 6){
            Debug.Log("敌人消失了！");
            Destroy(p);
        }
    }

    private void FixedUpdate()
    {
        
    }
}
```
---



## 二、SceneTest（场景测试）

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneTest : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        //场景类，场景管理类 

        //1.获取当前场景
        Scene scene = SceneManager.GetActiveScene();
        Debug.Log(scene.name);  //场景名称

        //2.场景是否已经加载
        Debug.Log(scene.isLoaded);

        //3.场景路径
        Debug.Log(scene.path);

        //4.场景索引
        Debug.Log(scene.buildIndex);
        GameObject[] gos = scene.GetRootGameObjects();
        Debug.Log(gos.Length);

        //场景管理类:

        //1.直接创建一个新场景
        Scene newScene = SceneManager.CreateScene("newScene");
        //2.已加载场景个数
        Debug.Log(SceneManager.sceneCount);
        //3.卸载场景
        SceneManager.UnloadSceneAsync(newScene);

        //加载场景

        //1.替换
        //SceneManager.LoadScene("MyScene",LoadSceneMode.Single);
        //2.添加（两个场景内容叠加）
        SceneManager.LoadScene("MyScene", LoadSceneMode.Additive);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
```
---



## 三、EmptyTest（空测试）

```c#
using JetBrains.Annotations;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EmptyTest : MonoBehaviour
{
    public GameObject Cube;

    //获取预设体
    public GameObject Prefab;

    // Start is called before the first frame update
    void Start()
    {
        //GameObject go=this.gameObject;
        Debug.Log(gameObject.name);

        //tag标签
        Debug.Log(gameObject.tag);

        //layer图层
        Debug.Log(gameObject.layer);

        //立方体的名称
        Debug.Log(Cube.name);
        //当前真正的激活状态
        Debug.Log(Cube.activeInHierarchy);
        //它自身的激活状态
        Debug.Log(Cube.activeSelf);

        //获取位置信息
        Debug.Log(transform.position);

        //获取其他组件
        BoxCollider bc = GetComponent<BoxCollider>();

        //添加一个组件
        Cube.AddComponent<AudioSource>();

        //通过预设体来实例化一个物体
        GameObject p = Instantiate(Prefab, Vector3.one, Quaternion.identity);
        Instantiate(Prefab, Vector3.zero, Quaternion.identity);

        //销毁物体
        Destroy(p);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
```
---



## 四、ApplicationTest（应用测试）

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ApplicationTest : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        //游戏数据文件夹路径（只读，加密压缩）
        Debug.Log(Application.dataPath);

        //持久化文件夹路径
        Debug.Log(Application.persistentDataPath);

        //StreamingAssets文件夹路径（只读，配置文件）
        Debug.Log(Application.streamingAssetsPath);

        //临时文件夹
        Debug.Log(Application.temporaryCachePath);

        //控制是否在后台运行
        Debug.Log(Application.runInBackground);

        //打开url（即直接跳转网站）
        //Application.OpenURL("https://gitee.com/qq2607563994/algorithm-code");

        //退出游戏
        Application.Quit();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
```
---



## 五、TransformTest（坐标测试）
```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TransformTest : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        //获取位置
        Debug.Log(transform.position);
        Debug.Log(transform.localPosition);

        //获取旋转
        Debug.Log(transform.rotation);
        Debug.Log(transform.localRotation);
        Debug.Log(transform.eulerAngles);
        Debug.Log(transform.localEulerAngles);

        //获取缩放
        Debug.Log(transform.localScale);

        //向量
        Debug.Log(transform.forward);
        Debug.Log(transform.right);
        Debug.Log(transform.up);
        

        //父子关系

        //获取父物体
        //GameObject t = transform.parent.gameObject;

        //子物体个数
        Debug.Log(transform.childCount);

        //解除父子关系
        transform.DetachChildren();

        //获取子物体
        Transform trans = transform.Find("Child");
        trans = transform.GetChild(0);

        //判断一个物体是不是另外一个物体的子物体
        bool res = transform.IsChildOf(transform);
    }

    // Update is called once per frame
    void Update()
    {
        //时刻看向原点
        transform.LookAt(Vector3.zero);

        //旋转
        transform.Rotate(Vector3.up, 0.5f);

        //绕某个物体旋转
        transform.RotateAround(Vector3.zero, Vector3.up, 0.2f);

        //移动
        //transform.Translate(Vector3.forward * 0.1f);
    }
}

```
---



## 六、KeyTest（按键测试）
```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class KeyTest : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        //鼠标点击
        //按下鼠标 0左键 1右键 2滚轮
        if(Input.GetMouseButtonDown(0))
        {
            Debug.Log("按下了鼠标左键");
        }
        //持续按下
        if (Input.GetMouseButton(0))
        {
            Debug.Log("持续按下了鼠标左键");
        }
        //抬起按键
        if (Input.GetMouseButtonUp(0))
        {
            Debug.Log("抬起了鼠标左键");
        }

        //键盘点击
        //按下
        if (Input.GetKeyDown(KeyCode.A))
        {
            Debug.Log("按下了A");
        }
        if (Input.GetKey(KeyCode.A))
        {
            Debug.Log("持续按下了A");
        }
        if (Input.GetKeyUp(KeyCode.A))
        {
            Debug.Log("抬起了A");
        }
    }
}

```
---



## 七、AsyncTest（异步加载测试）
```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class AsyncTest : MonoBehaviour
{
    AsyncOperation operation;

    void Start()
    {
        StartCoroutine(loadScene());
    }

    //协程方法用来异步加载场景
    IEnumerator loadScene()
    {
        operation = SceneManager.LoadSceneAsync("MyScene");

        //加载完场景不要自动跳转
        operation.allowSceneActivation = false;

        yield return operation;
    }

    float timer;

    void Update()
    {
        Debug.Log(operation.progress);
        timer += Time.deltaTime;

        //如果大于5秒后跳转
        if(timer > 5)
        {
            operation.allowSceneActivation = true;
        }
    }
}

```
---
