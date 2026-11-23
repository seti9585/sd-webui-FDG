# sd-webui-FDG

Frequency-decoupled guidance for reForge txt2img. Extract into extensions, restart reForge and enable FDG. Start with standard Euler, Hires.fix off, one prompt and one image. Low and High are absolute CFG scales.

## 日本語

ZIP内の sd-webui-FDG フォルダーを extensions 直下に置き、reForgeを再起動します。既存のreForge環境のPyTorchとKorniaを使用します。

txt2imgで標準Euler、Hires.fix OFF、batch size/count 1、通常の正負プロンプトを使用してください。NGMS、Skip Early CFG、epsilon scaling、他のCFG拡張はOFFにします。モデル・seed・steps・解像度を固定し、次の3枚を比較します。

1. FDG OFF、本体CFG 5
2. FDG OFF、本体CFG 10
3. FDG ON、本体CFG 10、Low 5、High 10

配線確認にはLowとHighを両方10にして2枚目と比較します。検証時は webui-user.bat の call webui.bat より前に set SD_WEBUI_SETI_DEBUG=1 を追加し、再起動してください。通常時はスキップ理由とエラーのみ表示します。デバッグ時のコンソールの Predictions returned が0、中断、エラーの画像は比較から除外してください。PNGの FDG requested は要求設定です。

## Dependencies and references / 依存関係・典拠

Uses the host installation of PyTorch, Kornia and Gradio. No automatic installation. The local compatibility implementation uses Kornia pyrdown/pyrup with explicit padding.

- [FDG paper](https://arxiv.org/abs/2506.19713)
- [Kornia](https://github.com/kornia/kornia)

