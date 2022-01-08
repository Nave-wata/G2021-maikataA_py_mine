# test_cython

## Cython完成

主にこれを使う（多分2つとも一緒）

```cmd
$ cythonize -b *.pyx
>>> カレントディレクトリに.cファイル・実行ファイル作成

$ cythonize -i *.pyx
>>> カレントディレクトリに.cファイル・実行ファイル作成
```

こういう方法もある

```cmd
$ cythonize *.pyx
>>> カレントディレクトリに.cファイル作成

$ cythonize -b *.c
>>> カレントディレクトリに実行ファイル作成
```

## 参考サイト

- [[Python3.7] windows10上の Jupyter Notebook で Cython を動かそうとしたら "DistutilsPlatformError: Unable to find vcvarsall.bat"](https://qiita.com/siruku6/items/0acd240df2e842300a30)
- [Cythonのコンパイルが簡単に！cythonizeコマンドを使おう](https://miyabikno-jobs.com/pc/cython-compile-cythonize/)
