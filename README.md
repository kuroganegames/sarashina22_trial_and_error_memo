# sarashina22_trial_and_error_memo

メモ周りは[zennのスクラップ](https://zenn.dev/kurogane/scraps/b61e976ff1d93f)に保存しています。

主に試行錯誤用のコードをコチラに。


## [check_tokenizer.ipynb](https://github.com/kuroganegames/sarashina22_trial_and_error_memo/blob/main/check_tokenizer.ipynb)

unslothにchat templateを追加する時のメモ。

unslothのコード内のgemma3部分に記載してあった<br/>
`print(tokenizer.chat_template.replace("}\n", "####").replace("\n", "\\n").replace("####", "}\n"))`<br/>
で作成。


## [sarashina22_(1B)_Conversational.ipynb](https://github.com/kuroganegames/sarashina22_trial_and_error_memo/blob/main/sarashina22_(1B)_Conversational.ipynb)

トレーニングが完走したnotebook
