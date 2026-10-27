# Adds the date of a search to the history of the searches

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("engine", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="userquery",
            name="created_on",
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
