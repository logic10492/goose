# shuorenhua для goose

`shuorenhua` — устанавливаемый Open Plugin для goose с skill, который убирает шаблонность и излишнюю искусственность из китайского и английского текста, сохраняя факты, термины, стиль и ответственность.

## Установка и активация

Установите плагин из этого каталога через существующий механизм Open Plugins goose, указав `plugin.json` как manifest. После установки skill доступен под именем `shuorenhua:shuorenhua`; используйте это имя в настройках или в рабочем процессе goose.

Описание skill в списке goose берётся из frontmatter файла `skills/shuorenhua/SKILL.md`, а не из поля `description` корневого `plugin.json`. Полный текст skill и supporting files из `references/` загружаются по требованию через `load_skill`, поэтому список остаётся кратким, а подробные правила доступны во время работы.

В пакет намеренно не входят `evals/` и `automation/`: это upstream-материалы для оценки и автоматизации, а не runtime-зависимости goose. Ссылки на `evals/` в upstream `SKILL.md` сохранены как материалы upstream для сопровождения и оценки; они не требуются для установки или выполнения skill.

## Upstream

Содержимое skill и references перенесено из `MrGeDiao/shuorenhua` версии `2.4.0` с закреплённого commit. Лицензия и подробности атрибуции находятся в `LICENSE` и `ATTRIBUTIONS.md`.
