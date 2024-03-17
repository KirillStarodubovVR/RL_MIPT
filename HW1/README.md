**Обучение на основе функции полезности**

Задача: воспроизвести результаты DQN на двух играх Atari.

Описание. Ваша задача — воспроизвести результаты DQN на играх Pong и другой игре на ваш выбор из таблицы лекций или семинара. Важно, чтобы вы не выбрали игру, где результат человека значительно превосходит DQN. Учтите, что некоторые игры могут потребовать специфических методов, например, MontezumaRevenge.

Цель состоит в том, чтобы ваш агент достиг среднего вознаграждения, указанного в таблице для *DQN*, с учетом std dev.

**Детали:**

* Воспользуйтесь техниками из оригинальной статьи DeepMind: *replay buffer*, *frame skip*, *target network*.
* Можно использовать любой фреймворк.
* Приветствуются улучшения базового кода или дополнительные техники.
  
**Что нужно сдать:**

* Код обучения в формате *Jupyter*.
* График сходимости, показывающий среднее вознаграждение.
* Веса обученной модели.
* Код для запуска модели с готовыми весами.
* Выводы по используемым гиперпараметрам, таким как размер *replay buffer*, *learning rate* и т. д.

Для решения задачи выбраны 2 среды: Pong и Breakout

Breakout:

![Breakout GIF](https://gymnasium.farama.org/_images/breakout.gif)

Информация о среде:
| Action Space |          Observation Space            |              Import               |
|:------------:|:-------------------------------------:|:---------------------------------:|
| Discrete(4)  |   Box(0, 255, (210, 160, 3), uint8)   | gymnasium.make("ALE/Breakout-v5") |

**Описание**

Еще одна знаменитая игра от Atari. По своей динамике она похожа на понг: Вы перемещаете пластину и бьете мячом в кирпичную стену в верхней части экрана. Ваша цель - разрушить кирпичную стену. Вы можете попытаться пробить стену и позволить мячу устроить хаос на другой стороне, причем самостоятельно! У вас есть пять жизней.

Pong:

![Pong GIF](https://gymnasium.farama.org/_images/pong.gif)

| Action Space |          Observation Space            |              Import               |
|:------------:|:-------------------------------------:|:---------------------------------:|
| Discrete(6)  | Box(0, 255, (210, 160, 3), uint8)     | gymnasium.make("ALE/Pong-v5") |

**Описание**

Вы управляете правым веслом и соревнуетесь с левым веслом, которым управляет компьютер. Каждый из вас старается отклонить мяч от своих ворот в ворота противника.

Архтектура модели DQN:
```python
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
```

Replay buffer используется от SB3
```python
    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
```

Гиперпараметры модели выбраны следующими 



| Game         | Num. Timesteps | learning rate  | buffer size   |     gamma    |       tau      |  target network frequency | batch size |
|:------------:|:--------------:|:--------------:|:-------------:|:------------:|:--------------:|:-------------------------:|:----------:|
| Breakout     | 10_000_000     | 1e-4           | 1_000_000     | 0.99         | 1.0            | 1000                      | 32         |
| Pong         | 10_000_000     | 1e-4           | 1_000_000     | 0.99         | 0.8            | 800                       | 32         |


Графики обучения среды Breakout:



Графики обучения среды Pong:

Результаты расчёта логировались с помощью wandb:
График длины эпизода показан ниже [OpenAI](https://openai.com)

