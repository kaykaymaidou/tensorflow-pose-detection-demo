import React, { useEffect } from 'react';
import './App.css';

/*
  多重姿态检测可以解码图像中的多个姿势. 比单个姿势检测算法复杂得多, 并且运行速度稍慢, 但却在图像中有多人的情况下很有优势, 检测到的关键点不太可能与错误的姿势相关联;
  即使用于检测单个人的姿势, 这种算法也可能更可取. 因为当多个人出现在图像中时, 两个姿势被连接在一起的意外就不会发生.
  它使用快速贪婪解码算法, 这个算法来自于 PersonLab 的研究论文: 使用 Bottom-Up, Part-Based 和 Geometric Embedding 模型的人体姿势检测和实例分割
 */

import * as PoseDetection from '@tensorflow-models/pose-detection'
import '@tensorflow/tfjs-backend-webgl'
import { PoseDetector } from '@tensorflow-models/pose-detection/dist/pose_detector';

// 这是一个纯函数, 因此需要用 hooks 来监听变化, 执行生命周期
function App() {
  // 动作输入(可能是视频, 可能是图片, 可能是动图等)
  let poseNetInput: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement
  // 检测输出对象
  let poseNetOutput: HTMLCanvasElement
  // 检测输出对象上下文对象
  let poseNetOutputCtx: CanvasRenderingContext2D
  // 姿势检测
  let poseDetector: PoseDetector
  // 姿势检测的模型
  /*
    MoveNet: 是一种速度快、准确率高的姿态检测模型, 可检测人体的 17 个关键点, 能够以 50+ fps 的速度在笔记本电脑和手机上运行
    BlazePose: MediaPipe BlazePose 可以检测人体 33 个关键点, 除了 17 个 COCO 关键点之外, 它还为脸部、手和脚提供了额外的关键点检测
    PoseNet: 可以检测多个姿态, 每个姿态包含 17 个关键点
   */
  let model: PoseDetection.SupportedModels.PoseNet
  // 创建一个输入对象的拷贝图像
  const poseNetInputCopy = document.createElement('canvas') as HTMLCanvasElement
  // 设置拷贝对象的宽高
  poseNetInputCopy.width = 480
  poseNetInputCopy.height = 480

  /**
   * 初始化方法
   */
  const init = async () => {
    // 姿势检测输出对象
    poseNetOutput = document.getElementById('canvas') as HTMLCanvasElement
    // 姿势检测输出对象的上下文对象
    // !! 后面感叹号是TS得语法, 表示类型推断排除 null 和 undefined
    poseNetOutputCtx = poseNetOutput.getContext('2d')!
    // 姿势检测输入对象
    poseNetInput = document.getElementById('video') as HTMLVideoElement
    // 这里是获取摄像头作动作输入源, 若有摄像头可以尝试这个, 还需要WebRTC进行数据传输
    // const stream = await navigator.mediaDevices.getUserMedia({
    //   audio: false,
    //   video: true
    // })
    // poseNetInput.srcObject = stream
    // 这里因为是PC端, 无摄像头, 因此用视频来模拟动作输入源
    poseNetInput.src = require('./assets/video/Pose-Mediapipe.mp4')
    // 选择姿势检测的检测模型
    model = PoseDetection.SupportedModels.PoseNet
    // 有三种模型的类型可供选择: "lite"、"full" 和 "heavy" 这些改变了检测模型的大小 Lite 的检测精度最低, 但性能最好, 而 "heavy" 的检测精度最好, 但更消耗性能, 而 "full" 则处于中间位置我们选择了它
    poseDetector = await PoseDetection.createDetector(model, {
      modelType: 'full',
    })
    // 姿势检测方法
    detectPose()
  }

  /**
   * 检测动作方法
   */
  const detectPose = async () => {
    // 获取输入对象拷贝的上下文对象
    const poseNetInputCopyCtx = poseNetInputCopy.getContext('2d')
    // 将当前输入图像绘制到拷贝对象里, 设置位置, 宽高保持和输入对象宽高一样
    poseNetInputCopyCtx?.drawImage(poseNetInput, 0, 0, poseNetInput.width, poseNetInput.height)
    // 将拷贝对象绘制到输出对象里
    poseNetOutputCtx.drawImage(poseNetInput, 0, 0, poseNetInputCopy.width, poseNetInputCopy.height)
    // 对拷贝出来的对象进行姿势预测, 因为使用的是poseNet, 因此每个姿势的关键点也就只是17个
    const poseArray = await poseDetector.estimatePoses(poseNetInputCopy, {
      // 默认为 false. 如果姿势应该进行水平的翻转/镜像. 对于视频默认水平翻转的视频(比如网络摄像头), 如果你希望姿势回到原来正确的方向, 改参数设置为 true
      flipHorizontal: false,
      // 最大可检测姿势数量, 默认 5
      maxPoses: 1,
      // (置信度)只返回根部位得分大于或等于这个值的检测实例, 默认为 0.5
      scoreThreshold: 0.5,
      // Non-maximum 抑制部位距离, 它必须为正. 如果两个部位距离小于 nmsRadius 像素, 就会相互抑制. 默认为 20
      nmsRadius: 20
    })
    // 列出所有的点集合, 因为我们最大检测人数是1, 因此从poseArray里面获取数组第一个元素就是我们想要的
    const pointList = poseArray[0]?.keypoints || []

    // 获取相邻的关键点信息
    const adjacentPairs = PoseDetection.util.getAdjacentPairs(model)
    // 画出所有连线
    adjacentPairs.forEach(([i, j]: any) => {
      // 一个姿势检测关键点
      const kp1 = pointList[i]
      // 与上一个关键点相邻的点
      const kp2 = pointList[j]
      // score 不为空就画线
      const score1 = kp1.score != null ? kp1.score : 1
      const score2 = kp2.score != null ? kp2.score : 1
      // 只绘制置信度大于等于0.5的点位的线段
      if (score1 >= 0.5 && score2 >= 0.5) {
        // 画骨架(就是把关键点连起来)
        drawSkeleton(poseNetOutputCtx, [kp1.x, kp1.y], [kp2.x, kp2.y], '#40e0d0', 2)
      }
    })

    // 画出所有关键点(17个关键点)
    pointList.forEach(({ x, y, score, name }: any) => {
      // 只绘制置信度大于等于0.5的点位
      if (score > 0.5) {
        // 画关键点
        drawKeyPoint(poseNetOutputCtx, x, y, 5, '#f9274c')
      }
    })

    // 使用60fps可能在人肉眼的观测下可能会卡, 因为目前大多数屏幕都还是60fps
    // requestAnimationFrame(() => detectPose())
    // 因此用setTimeout去设置渲染的时间
    const timer = setTimeout(() => {
      clearTimeout(timer)
      detectPose()
    }, 50)
  }

  /**
   * 画关键点(实心圆)
   * @param ctx 绘制对象
   * @param x 该点位的横坐标
   * @param y 该点位的纵坐标
   * @param r 实心圆的半径
   * @param circleColor 实心圆的颜色
   */
  function drawKeyPoint(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, circleColor: string) {
    ctx.beginPath()
    ctx.arc(x, y, r, 0, 2 * Math.PI)
    ctx.fillStyle = circleColor
    ctx.fill()
  }

  /**
   * 画骨架(就是把关键点连起来)
   * @param ctx 绘制对象
   * @param param0 第一个检测关键点
   * @param param1 与第一个检测关键点相邻的点位
   * @param lineColor 连线的颜色
   * @param lineWidth 连线线的宽度
   */
  function drawSkeleton(ctx: CanvasRenderingContext2D, [ax, ay]: number[], [bx, by]: number[], lineColor: string, lineWidth: number) {
    ctx.beginPath()
    ctx.moveTo(ax, ay)
    ctx.lineTo(bx, by)
    ctx.strokeStyle = lineColor
    ctx.lineWidth = lineWidth
    ctx.stroke()
  }

  // 因为 useEffect 的回调函数要是同步的, 因此需要再包一层然后才能调用回调
  useEffect(() => {
    const request = async () => {
      await init()
    }
    request()
  })

  return (
    <div className="App">
      <div className="input">
        <video id="video" className="input-source" width="480" height="480" autoPlay playsInline muted loop></video>
      </div>
      <div className="output">
        <canvas id="canvas" className='output-container' width="480" height="480"></canvas>
      </div>
    </div>
  );
}

export default App;
