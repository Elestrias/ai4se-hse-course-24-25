# AI. Лабараторная №2
## Гильванов Руслан Маратович

## Решение
Для исследования использовал два языка: python и java. Решение находится полностью в файле Solution.ipynb \
Сначала я обработал датасет: с помощью Tree-sitter извлек тело, имя функции и все комментарии (однострочные в случае обоих языков, docstring (для python) и многострочные комментарии (для java)). В датасет записывал тело функции, спаршенное tree-sitter, имя функции и тело без комментариев (брал оригинальное тело и удалял все встреченные коментарии). Так же была произведена дополнительная постобработка тел функций с комментариями и без путем добавления 
```
f'def <extra_id_0> :{body}'
```
в случае python и 
```
f'public final void <extra_id_0> {body}'
```
в случае с java.

Следовательно для python и java была примерная одинаковая обработка датасета, различия были в Query для триситтера и постобработке. В оригинальном датасете имя полей совпадало. Еще есть различия в постобработке, синтаксис функции в java и python разные

На этапе с моделью для python и java происходило все одинаково. Был подобран max_length=80 на датасете с python. На java этот же max_length=80 отработал в ожидаемых метриках

## Метрики

| Language          | exact_match  | rouge1             |
| ----------------- | ------------ | ------------------ |
| Python + comments | 0.202        | 0.4868611111111113 |
| Python            | 0.137        | 0.3458994033744034 |
| Java + comments   | 0.3          | 0.3026666666666667 |
| Java              | 0.298        | 0.3006666666666667 |
 

## Примеры извлеченных функций, тел функций и тел функций без комментариев
### Java original
```
    @CheckReturnValue
    @SchedulerSupport(SchedulerSupport.CUSTOM)
    public final Observable<T> skipLast(long time, TimeUnit unit, Scheduler     scheduler, boolean delayError, int bufferSize) {
        ObjectHelper.requireNonNull(unit, "unit is null");
        ObjectHelper.requireNonNull(scheduler, "scheduler is null");
        ObjectHelper.verifyPositive(bufferSize, "bufferSize");
        // the internal buffer holds pairs of (timestamp, value) so double the default buffer size
        int s = bufferSize << 1;
        return RxJavaPlugins.onAssembly(new ObservableSkipLastTimed<T>(this, time, unit, scheduler, s, delayError));
    }
```
### Java name
```
skipLast
```
### Java body with comments
```
public final void <extra_id_0> {
        ObjectHelper.requireNonNull(unit, "unit is null");
        ObjectHelper.requireNonNull(scheduler, "scheduler is null");
        ObjectHelper.verifyPositive(bufferSize, "bufferSize");
        // the internal buffer holds pairs of (timestamp, value) so double the default buffer size
        int s = bufferSize << 1;
        return RxJavaPlugins.onAssembly(new ObservableSkipLastTimed<T>(this, time, unit, scheduler, s, delayError));
    }
```

### Java body without comments
```
public final void <extra_id_0> {
        ObjectHelper.requireNonNull(unit, "unit is null");
        ObjectHelper.requireNonNull(scheduler, "scheduler is null");
        ObjectHelper.verifyPositive(bufferSize, "bufferSize");
        
        int s = bufferSize << 1;
        return RxJavaPlugins.onAssembly(new ObservableSkipLastTimed<T>(this, time, unit, scheduler, s, delayError));
    }
```

### Python original
```
def get_vid_from_url(url):
        """Extracts video ID from URL.
        """
        return match1(url, r'youtu\.be/([^?/]+)') or \
          match1(url, r'youtube\.com/embed/([^/?]+)') or \
          match1(url, r'youtube\.com/v/([^/?]+)') or \
          match1(url, r'youtube\.com/watch/([^/?]+)') or \
          parse_query_param(url, 'v') or \
          parse_query_param(parse_query_param(url, 'u'), 'v')
```
### Python name
```
get_vid_from_url
``` 
### Python body with comments
```
def <extra_id_0> :"""Extracts video ID from URL.
        """
        return match1(url, r'youtu\.be/([^?/]+)') or \
          match1(url, r'youtube\.com/embed/([^/?]+)') or \
          match1(url, r'youtube\.com/v/([^/?]+)') or \
          match1(url, r'youtube\.com/watch/([^/?]+)') or \
          parse_query_param(url, 'v') or \
          parse_query_param(parse_query_param(url, 'u'), 'v')
```
### Python body without comments
```
def <extra_id_0> :
        return match1(url, r'youtu\.be/([^?/]+)') or \
          match1(url, r'youtube\.com/embed/([^/?]+)') or \
          match1(url, r'youtube\.com/v/([^/?]+)') or \
          match1(url, r'youtube\.com/watch/([^/?]+)') or \
          parse_query_param(url, 'v') or \
          parse_query_param(parse_query_param(url, 'u'), 'v')
```

## Примеры плохих метрик
### Java
Original code:
```
protected final void fastPathOrderedEmit(U value, boolean delayError, Disposable disposable) {
        final Observer<? super V> observer = downstream;
        final SimplePlainQueue<U> q = queue;

        if (wip.get() == 0 && wip.compareAndSet(0, 1)) {
            if (q.isEmpty()) {
                accept(observer, value);
                if (leave(-1) == 0) {
                    return;
                }
            } else {
                q.offer(value);
            }
        } else {
            q.offer(value);
            if (!enter()) {
                return;
            }
        }
        QueueDrainHelper.drainLoop(q, observer, delayError, disposable, this);
    }
```

Предсказание (с комментами и без):
```
accept
```
### Python
Original code: 
```
def get_vid_from_url(url):
        """Extracts video ID from URL.
        """
        return match1(url, r'youtu\.be/([^?/]+)') or \
          match1(url, r'youtube\.com/embed/([^/?]+)') or \
          match1(url, r'youtube\.com/v/([^/?]+)') or \
          match1(url, r'youtube\.com/watch/([^/?]+)') or \
          parse_query_param(url, 'v') or \
          parse_query_param(parse_query_param(url, 'u'), 'v')
```

Предсказание без комментов
```
match1
```

Предсказание с комментами
```
or
```

# Выводы
Модели отрабатывают в задданных промежутках, следовательно можно утверждать, что задание успешно выполнено